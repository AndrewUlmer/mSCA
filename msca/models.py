from typing import Union
import numpy as np
from tqdm import tqdm

from .coordinated_dropout import *
from .initializations import _initialize
from .architectures import *


# from msca import *
from .loss_funcs import *
from .utils import *

# from msca.evaluations import bootstrap_performances


def convert_to_dataloader(X, batch_size=64, shuffle=True):
    """
    Utility function for converting to PyTorch dataloader

    Parameters
    ----------
    X : dict
        Formatting described in quickstart.ipynb

    Returns
    -------
    output : PyTorch dataloader object
    """

    # Convert to torch tensors
    X_tensor = torchify(X)

    # Pad trials to be the same length
    X_padded, trial_lengths = pad_trials(X_tensor)

    # Convert X_padded to list of dicts
    X_list_of_dicts = to_list_of_dicts(X_padded)

    # Store trial lengths + tensors
    X_named_tuples = to_named_tuples(X_list_of_dicts, trial_lengths)

    # Convert to data loader
    train_loader = torch.utils.data.DataLoader(
        X_named_tuples, batch_size=batch_size, shuffle=shuffle  # type: ignore
    )
    return train_loader, trial_lengths


class mSCA:
    """
    The interface for mSCA: training, latent inference, and prediction

    n_components : int
        Number of latent dimensions.
    n_epochs : int, optional
        Number of training epochs (default: 8000).
    loss_func : str
        Loss function to use: 'Gaussian' (for pre-smoothed data) or 'Poisson' (for binned data).
    lr : float, optional
        Learning rate for training (default: 1e-3).
    filter_len : int, optional
        Length of smoothing filter (default: 21).
    linear : bool, optional
        If True, use a linear encoder; if False, use a nonlinear encoder (default: False).
    region_weights : dict, optional
        Per-region weights for reconstruction loss. Keys are region names.
    lam_sparse : float, optional
        Weight for L1 sparsity loss. If None, defaults to 10% of reconstruction loss for Gaussian model;
        4% for the Poisson model
    lam_orthog : float, optional
        Weight for orthogonality loss. If None, defaults to 10% of reconstruction loss for Gaussian model;
        1% for the Poisson model
    lam_region : float, optional
        Weight for region-sparsity loss. If None, defaults to 0%.
    batch_size : int, optional
        Batch size for training (default: 64).
    cd_rate : float, optional
        Coordinated dropout rate (default: 0.0).

    TODO:
        - Confirm defaults for hyperparameters (lam_x) work properly
        - Finalize defaults for lam_sparse (Gaussian and Poisson)
        - Finalize default for lam_region (Gaussian and Poisson)
        - Add GPU support
    """

    def __init__(
        self,
        n_components,
        n_epochs: int = 8000,
        loss_func: str = "Gaussian",
        lr: float = 1e-3,
        filter_len: int = 21,
        linear: bool = False,
        region_weights: Union[None, list[np.ndarray]] = None,
        lam_sparse: Union[None, float] = None,
        lam_orthog: Union[None, float] = None,
        lam_region: Union[None, float] = None,
        batch_size: int = 64,
        cd_rate: float = 0.5,
        device: str = "cpu",
    ):
        self.n_components = n_components
        self.n_epochs = n_epochs
        self.loss_func = loss_func
        self.lr = lr
        self.filter_len = filter_len
        self.linear = linear
        self.region_weights = region_weights
        self.lam_sparse = lam_sparse
        self.lam_orthog = lam_orthog
        self.lam_region = lam_region
        self.batch_size = batch_size
        self.cd_rate = cd_rate
        self.device = device

    def fit(
        self, X: dict[str, list[np.ndarray]]
    ) -> tuple[object, dict[str, np.ndarray]]:
        #### POST-HOC SCALING
        from msca.evaluations import bootstrap_performances

        # Store the names of the regions
        self.region_names = list(X.keys())

        # Compute initial model parameters
        (
            init_encoder,
            init_decoder,
            init_decoder_bias,
            auto_lam_sparse,
            auto_lam_orthog,
            auto_lam_region,
            auto_region_weights,
            region_sizes,
        ) = _initialize(
            X,
            self.n_components,
            self.loss_func,
            self.lam_sparse,
            self.lam_orthog,
            self.lam_region,
        )

        # Set hyperparameters to be used during training
        self.region_weights = (
            auto_region_weights if self.region_weights is None else self.region_weights
        )
        print(f"Using region-weights = {auto_region_weights}")

        ##### TESTING POST-HOC SCALING
        self.pre_lam_sparse = self.lam_sparse

        # Automatically set lam_sparse
        self.lam_sparse = (
            auto_lam_sparse if self.cd_rate == 0.0 else auto_lam_sparse * self.cd_rate
        )
        print(f"Using lam_sparse = {self.lam_sparse}")

        # Automatically set the orthogonality penalty
        self.lam_orthog = (
            auto_lam_orthog if self.cd_rate == 0.0 else auto_lam_orthog * self.cd_rate
        )
        print(f"Using lam_orthog = {self.lam_orthog}")

        # Region-sparsity loss
        self.lam_region = (
            auto_lam_region if self.cd_rate == 0.0 else auto_lam_region * self.cd_rate
        )
        print(f"Using lam_region = {self.lam_region}")

        # Instantiate model architecture and set inital params
        self.model = mSCA_architecture(
            init_encoder,
            init_decoder,
            init_decoder_bias,
            region_sizes,
            self.n_components,
            linear=self.linear,
            loss_func=self.loss_func,
        )

        # Convert input data to dataloader TODO: remove non-shuffling (for testing purposes)
        data_loader, _ = convert_to_dataloader(
            X, batch_size=self.batch_size, shuffle=False
        )

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        #### TESTING POST-HOC SCALING
        self.optimizer_post_hoc = torch.optim.Adam(
            [{"params": self.model.C, "lr": 1e-3}]
        )

        # Make coordinated dropout object
        self.cd = CoordinatedDropout(self.n_components, self.cd_rate, self.filter_len)

        # Used for truncating trials
        self.trunc = slice(self.filter_len - 1, -self.filter_len + 1)
        self.half_trunc = slice(self.filter_len // 2, -(self.filter_len // 2))

        # Set up loss function
        train_criterion = eval(f"{self.loss_func}_loss".lower())

        # Initialize loss tracking
        train_loss_dicts = {
            "reconstruction": [],
            "latent_sparsity": [],
            "region_sparsity": [],
            "orthogonality": [],
        }

        # Iterate through training loop
        for epoch in tqdm(range(self.n_epochs)):

            ### TESTING POST-HOC SCALING
            if epoch < self.n_epochs - 1000:

                # Training step
                self.criterion = train_criterion
                _, _, loss_dict = self.loop(data_loader, mode="train")

            elif epoch == self.n_epochs - 1000:
                pre_scale_perf = bootstrap_performances(self, X)
                torch.save(
                    pre_scale_perf,
                    f"./experiments/simulation/sparsity_sweep_decoder_single_trial/pre_{self.pre_lam_sparse:.4f}.pt",
                )

            else:
                self.model.mode = "post-hoc-scaling"
                _, _, loss_dict = self.loop(data_loader, mode="post-hoc-scaling")
                print(self.model.C)

            # Store training loss
            train_loss_dicts["reconstruction"].append(loss_dict["reconstruction"])
            train_loss_dicts["latent_sparsity"].append(loss_dict["latent_sparsity"])
            train_loss_dicts["region_sparsity"].append(loss_dict["region_sparsity"])
            train_loss_dicts["orthogonality"].append(loss_dict["orthogonality"])

        # Concatenate training losses over all epochs
        train_loss_dicts = {k: np.array(v) for k, v in train_loss_dicts.items()}

        #### TESTING: POST-HOC SCALING
        post_scale_perf = bootstrap_performances(self, X)
        torch.save(
            post_scale_perf,
            f"./experiments/simulation/sparsity_sweep_decoder_single_trial/post_{self.pre_lam_sparse:.4f}.pt",
        )

        return self, train_loss_dicts

    def loop(
        self, data_loader: torch.utils.data.DataLoader, mode: str = "train"
    ) -> tuple[dict[str, list], dict[str, list], dict[str, float]]:

        # Initialize containers for latents
        latents = {k: [] for k in self.region_names}
        reconstructions = {k: [] for k in self.region_names}

        # Object we will use to store training loss for this epoch
        epoch_loss_dict = {
            "reconstruction": 0.0,
            "latent_sparsity": 0.0,
            "region_sparsity": 0.0,
            "orthogonality": 0.0,
        }

        # Iterate over trials in the data_loader
        for _, (X_target, trial_length) in enumerate(data_loader):
            # Apply the mask to the inputs and outputs
            X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                self.cd.forward(
                    X_target,
                    trial_length,
                )
            )

            # Forward pass
            Z, Z_r, X_reconstruction = self.model(X_input_masked)

            # Apply the output mask to the reconstructions
            X_reconstruction_masked = self.cd.mask(
                X_reconstruction, truncate(output_mask, self.trunc)
            )

            # Mask the latents to account for trial length
            Z_masked = self.cd.mask(Z, truncate(Z_mask, self.half_trunc))

            if mode == "train":
                # Compute the loss
                loss, loss_dict = self.criterion(
                    X_reconstruction_masked,
                    truncate(X_output_masked, self.trunc),
                    self.region_weights,
                    Z_masked,
                    self.lam_sparse,
                    self.lam_region,
                    self.lam_orthog,
                    self.model.decoder_scaling,
                    self.model.decoder.model.weight,
                    mode=mode,
                )

                # Backpropagation and optimizer step (if training)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            #### TESTING POST-HOC SCALING
            elif mode == "post-hoc-scaling":
                r_loss_f = eval(self.loss_func.lower() + "_f")
                r_loss = sum(
                    reconstruction_loss(
                        X_reconstruction_masked,
                        truncate(X_output_masked, self.trunc),
                        r_loss_f,
                        mode="train",
                    )
                )
                r_loss.backward()
                self.optimizer_post_hoc.step()
                self.optimizer_post_hoc.zero_grad(set_to_none=True)

            # Accumulate loss
            if mode == "train":
                epoch_loss_dict["reconstruction"] += loss_dict["reconstruction"]
                epoch_loss_dict["latent_sparsity"] += loss_dict["latent_sparsity"]
                epoch_loss_dict["region_sparsity"] += loss_dict["region_sparsity"]
                epoch_loss_dict["orthogonality"] += loss_dict["orthogonality"]

            # Store latents
            elif mode == "evaluate":
                # Mask the post-convolution latents for visualization
                Z_r_masked = self.cd.mask(Z_r, truncate(Z_r_mask, self.trunc))

                # Append to latents
                [latents[k].append(Z_r_masked[k].squeeze()) for k in self.region_names]  # type: ignore

                # Append reconstructions
                [
                    reconstructions[k].append(X_reconstruction_masked[k].squeeze())  # type: ignore
                    for k in self.region_names
                ]

        return latents, reconstructions, epoch_loss_dict

    @torch.no_grad()
    def transform(
        self, X: dict[str, list[np.ndarray]], mode: str = "evaluate"
    ) -> dict[str, list[np.ndarray]]:
        # Convert inputs to data loader maintaining trial ordering
        data_loader, trial_lengths = convert_to_dataloader(
            X, batch_size=1, shuffle=False
        )

        # IMPORTANT: Disable coordinated dropout for finding latents
        self.cd.cd_rate = 0.0

        # Run model to get latents
        latents, _, _ = self.loop(data_loader, mode=mode)

        # Convert latents to numpy array
        latents = {k: [x.numpy() for x in v] for k, v in latents.items()}

        # Compute the magnitude of the latents within each region
        loadings = self.model.decoder.r_loadings()
        latents = {k: [z_i * loadings[k] for z_i in v] for k, v in latents.items()}

        # Remove the padding from the latents
        trial_lengths = np.array(trial_lengths) - (self.filter_len - 1) * 2
        latents = {
            k: [x[:t] for x, t in zip(v, trial_lengths)] for k, v in latents.items()
        }

        # Return latents
        return latents

    @torch.no_grad()
    def predict(
        self, X: dict[str, list[np.ndarray]], mode: str = "evaluate"
    ) -> dict[str, list[np.ndarray]]:
        # Convert inputs to data loader
        data_loader, trial_lengths = convert_to_dataloader(X, batch_size=1)

        # IMPORTANT: Disable coordinated dropout for finding latents
        self.cd.cd_rate = 0.0

        # Run model to get latents
        _, reconstructions, _ = self.loop(data_loader, mode=mode)

        # Convert latents to numpy array
        reconstructions = {
            k: [x.numpy() for x in v] for k, v in reconstructions.items()
        }

        # Remove the padding from the latents
        trial_lengths = np.array(trial_lengths) - (self.filter_len - 1) * 2
        reconstructions = {
            k: [x[:t] for x, t in zip(v, trial_lengths)]
            for k, v in reconstructions.items()
        }

        # Return latents
        return reconstructions

    def save(self, f: str):
        """
        Method for saving trained model weights in mSCA

        Parameters
        ----------
        f : str
            Path to save weights to
        """
        torch.save(self.model.state_dict(), f)

    def load(self, f: str):
        """
        Method for loading trained model weights in mSCA

        Parameters
        ----------
        f : str
            Path to load weights from
        """
        self.model.load_state_dict(torch.load(f))
