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

r_grads_all = []
l1_grads_all = []

sims = []


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
    decoder_type : str, optional
        Which decoder to use: "linear" (default), "nonlinear", or "triple_layer".
    decoder_hidden_size : int, optional
        Hidden size for the nonlinear decoder (default: 128).
    decoder_activation : str, optional
        Activation for the nonlinear decoder (default: "GeLU").
    decoder_init_mode : str, optional
        Initialization mode for nonlinear/triple_layer decoder: "pca" (default) or "random".

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
        lam_sparse: Union[None, float, str] = "adaptive",
        lam_orthog: Union[None, float] = None,
        lam_region: Union[None, float] = None,
        batch_size: int = 64,
        cd_rate: float = 0.5,
        decoder_type: str = "linear",
        decoder_hidden_size: int = 128,
        decoder_activation: str = "GeLU",
        decoder_init_mode: str = "pca",
        device: str = "cpu",
        cd_mode: str = "both",
        post_hoc_epoch: int = 1000,
        balance_interval: int = 100,
        init: str = "shared",
        grad_log_interval: int = 100,
        lam_sparse_ema: float = 0.3,
        lam_sparse_max: float = 50.0,
        sparsity_warmup_epochs: int = 0,
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
        self.decoder_type = decoder_type
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_activation = decoder_activation
        self.decoder_init_mode = decoder_init_mode
        self.cd_mode = cd_mode
        self.post_hoc_epoch = post_hoc_epoch
        self.balance_interval = balance_interval
        self.init = init
        self.grad_log_interval = grad_log_interval
        # EMA weight for adaptive lam_sparse update (0 = no update, 1 = no smoothing)
        self.lam_sparse_ema = lam_sparse_ema
        # Hard upper bound on adaptive lam_sparse to prevent runaway sparsification
        self.lam_sparse_max = lam_sparse_max
        # Number of epochs to train with no sparsity before enabling it
        self.sparsity_warmup_epochs = sparsity_warmup_epochs
        # print all the lam_x values
        # print(f"Initialized mSCA with lam_sparse={lam_sparse}, lam_orthog={lam_orthog}, lam_region={lam_region}")
    def fit(
        self, X: dict[str, list[np.ndarray]], load: bool = False
    ) -> tuple[object, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.region_names = list(X.keys())

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
            X, self.n_components, self.loss_func,
            self.lam_sparse, self.lam_orthog, self.lam_region, self.init,
        )

        self.region_weights = (
            auto_region_weights if self.region_weights is None else self.region_weights
        )
        print(f"Using region-weights = {auto_region_weights}")

        if self.lam_sparse == "adaptive":
            self.lam_sparse_mode = "adaptive"
        else:
            self.lam_sparse = (
                auto_lam_sparse if self.cd_rate == 0.0 else auto_lam_sparse * self.cd_rate
            )
            self.lam_sparse_mode = "fixed"
        print(f"Using lam_sparse = {self.lam_sparse}")

        self.lam_orthog = (
            auto_lam_orthog if self.cd_rate == 0.0 else auto_lam_orthog * self.cd_rate
        )
        print(f"Using lam_orthog = {self.lam_orthog}")

        self.lam_region = (
            auto_lam_region if self.cd_rate == 0.0 else auto_lam_region * self.cd_rate
        )
        print(f"Using lam_region = {self.lam_region}")

        self.model = mSCA_architecture(
            init_encoder, init_decoder, init_decoder_bias,
            region_sizes, self.n_components,
            linear=self.linear,
            loss_func=self.loss_func,
            filter_length=self.filter_len,
            decoder_type=self.decoder_type,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_activation=self.decoder_activation,
            decoder_init_mode=self.decoder_init_mode,
        )

        data_loader, _ = convert_to_dataloader(X, batch_size=self.batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.cd = CoordinatedDropout(self.n_components, self.cd_rate, self.filter_len, mode=self.cd_mode)

        self.trunc = slice(self.filter_len - 1, -self.filter_len + 1)
        self.half_trunc = slice(self.filter_len // 2, -(self.filter_len // 2))

        train_criterion = eval(f"{self.loss_func}_loss".lower())
        r_loss_f = eval(self.loss_func.lower() + "_f")    # raw loss fn for grad norm logging

        train_loss_dicts = {
            "reconstruction": [], "latent_sparsity": [],
            "region_sparsity": [], "orthogonality": [], "total_loss": [],
        }

        # Lambda value actually used each epoch (0.0 during warmup)
        train_lambda_dicts = {"lam_sparse": [], "lam_region": [], "lam_orthog": []}

        # Gradient norms of reconstruction and L1 losses w.r.t. encoder, each epoch
        train_gradient_dicts = {"r_grad": [], "l1_grad": []}

        if not load:
            self.criterion = train_criterion

            for epoch in tqdm(range(self.n_epochs)):

                # ── Phase 1: Warmup ── reconstruction only, sparsity zeroed ──────
                if epoch < self.sparsity_warmup_epochs:
                    _, _, loss_dict = self.loop(data_loader, mode="warmup")
                    lam_s_used, lam_r_used = 0.0, 0.0

                # ── Phase 2: Main training ── all losses active ───────────────────
                elif epoch < self.n_epochs - self.post_hoc_epoch:
                    if (
                        self.lam_sparse_mode == "adaptive"
                        and (epoch % self.balance_interval == 0
                            or isinstance(self.lam_sparse, str))
                    ):
                        # # if possion function and nonlinear decoder then times 0.1 or times 1
                        # if self.loss_func.lower() == "poisson" and self.decoder_type.lower() == "nonlinear":
                        #     self.lam_sparse = float(self.loop(data_loader, mode="balance-sparsity")) * 0.5
                        # else:
                        #     self.lam_sparse = float(self.loop(data_loader, mode="balance-sparsity"))
                        self.lam_sparse = float(self.loop(data_loader, mode="balance-sparsity"))

                    _, _, loss_dict = self.loop(data_loader, mode="train")
                    lam_s_used = float(self.lam_sparse) if not isinstance(self.lam_sparse, str) else 0.0
                    lam_r_used = float(self.lam_region)

                # ── Phase 3: Post-hoc scaling ─────────────────────────────────────
                elif epoch == self.n_epochs - self.post_hoc_epoch:
                    self._init_post_hoc()
                    _, _, loss_dict = self.loop(data_loader, mode="post-hoc-scaling")
                    lam_s_used, lam_r_used = 0.0, 0.0
                else:
                    _, _, loss_dict = self.loop(data_loader, mode="post-hoc-scaling")
                    lam_s_used, lam_r_used = 0.0, 0.0

                # Record lambda values used this epoch
                train_lambda_dicts["lam_sparse"].append(lam_s_used)
                train_lambda_dicts["lam_region"].append(lam_r_used)
                train_lambda_dicts["lam_orthog"].append(float(self.lam_orthog))

                # Record encoder gradient norms for reconstruction and L1 losses
                r_grad_norm, l1_grad_norm = self._compute_grad_norms(data_loader, r_loss_f)
                train_gradient_dicts["r_grad"].append(r_grad_norm)
                train_gradient_dicts["l1_grad"].append(l1_grad_norm)

                # Store epoch losses
                for k in ["reconstruction", "latent_sparsity", "region_sparsity",
                        "orthogonality", "total_loss"]:
                    train_loss_dicts[k].append(loss_dict[k])

                sims.append(self.model.decoder_scaling.clone().detach().numpy())

            # Finalize: convert to numpy arrays and store on self for save()
            train_loss_dicts = {k: np.array(v) for k, v in train_loss_dicts.items()}
            self.train_lambda_dicts = {k: np.array(v) for k, v in train_lambda_dicts.items()}
            self.train_gradient_dicts = {k: np.array(v) for k, v in train_gradient_dicts.items()}

            return self, train_loss_dicts, self.train_lambda_dicts, self.train_gradient_dicts

        else:
            return self  # type: ignore


    def _decoder_weight_for_loss(self) -> torch.Tensor:
        """
        Returns the decoder matrix used for orthogonality regularization.

        - linear decoder: decoder.model.weight
        - nonlinear decoder: effective decoder matrix output_layer @ hidden_layer
        - triple_layer decoder: effective decoder matrix layer3 @ layer2 @ layer1
        """
        if self.decoder_type.lower() == "linear":
            return self.model.decoder.model.weight
        elif self.decoder_type.lower() == "triple_layer":
            return self.model.decoder.layer3.weight @ self.model.decoder.layer2.weight @ self.model.decoder.layer1.weight
        else:  # nonlinear
            hidden_w = self.model.decoder.hidden_layer.weight
            output_w = self.model.decoder.output_layer.weight
            return output_w @ hidden_w

    def _all_param_grad_norm(self) -> float:
        return sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5

    def _log_grad_norms(self, data_loader) -> dict[str, float]:
        """
        Computes the L2 norm of the gradient of each scaled loss component
        w.r.t. all model parameters, averaged over batches. Used to diagnose
        which loss term dominates the parameter update at a given epoch.
        """
        r_loss_f = eval(self.loss_func.lower() + "_f")
        # self.lam_sparse may be a numpy scalar or 0-d torch.Tensor — float() handles all
        lam_s = float(self.lam_sparse) if not isinstance(self.lam_sparse, str) else 0.0
        keys = ["reconstruction", "latent_sparsity", "region_sparsity", "orthogonality"]
        norms = {k: [] for k in keys}

        for _, (X_target, trial_length) in enumerate(data_loader):
            X_input_masked, X_output_masked, output_mask, Z_mask, _ = self.cd.forward(
                X_target, trial_length
            )
            Z, _, X_reconstruction = self.model(X_input_masked)
            X_reconstruction_masked = self.cd.mask(
                X_reconstruction, truncate(output_mask, self.trunc)
            )
            Z_masked = self.cd.mask(Z, truncate(Z_mask, self.half_trunc))
            V = self._decoder_weight_for_loss()

            rc_list = reconstruction_loss(
                X_reconstruction_masked,
                truncate(X_output_masked, self.trunc),
                r_loss_f,
                mode="train",
            )
            scaled_losses = [
                sum(r * w for r, w in zip(rc_list, self.region_weights.values())),
                torch.sum(torch.abs(Z_masked)) * lam_s,
                region_sparsity_loss(Z_masked, self.model.decoder_scaling) * self.lam_region,
                torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2 * self.lam_orthog,
            ]

            for i, (key, loss) in enumerate(zip(keys, scaled_losses)):
                loss.backward(retain_graph=(i < len(scaled_losses) - 1))
                norms[key].append(self._all_param_grad_norm())
                self.optimizer.zero_grad(set_to_none=True)
        return {k: float(np.mean(v)) for k, v in norms.items()}

    def loop(
        self, data_loader: torch.utils.data.DataLoader, mode: str = "train"
    ) -> tuple:
        # Initialize containers for latents
        latents = {k: [] for k in self.region_names}
        reconstructions = {k: [] for k in self.region_names}

        # Object we will use to store training loss for this epoch
        epoch_loss_dict = {
            "reconstruction": 0.0,
            "latent_sparsity": 0.0,
            "region_sparsity": 0.0,
            "orthogonality": 0.0,
            "total_loss": 0.0,
        }

        r_loss_f = eval(self.loss_func.lower() + "_f")

        # Accumulators for balance-sparsity: collect norm per batch, average after loop
        if mode == "balance-sparsity":
            r_grads, l1_grads = [], []

        for _, (X_target, trial_length) in enumerate(data_loader):
            X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                self.cd.forward(X_target, trial_length)
            )

            Z, Z_r, X_reconstruction = self.model(X_input_masked)

            X_reconstruction_masked = self.cd.mask(
                X_reconstruction, truncate(output_mask, self.trunc)
            )
            Z_masked = self.cd.mask(Z, truncate(Z_mask, self.half_trunc))

            # ── Warmup: train with sparsity penalties zeroed ──────────────────
            if mode == "warmup":
                loss, loss_dict = self.criterion(
                    X_reconstruction_masked,
                    truncate(X_output_masked, self.trunc),
                    self.region_weights,
                    Z_masked,
                    0.0,               # lam_sparse zeroed
                    0.0,               # lam_region zeroed
                    self.lam_orthog,
                    self.model.decoder_scaling,
                    self._decoder_weight_for_loss(),
                    mode="train",
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            # ── Train: full losses active ─────────────────────────────────────
            elif mode == "train":
                loss, loss_dict = self.criterion(
                    X_reconstruction_masked,
                    truncate(X_output_masked, self.trunc),
                    self.region_weights,
                    Z_masked,
                    self.lam_sparse,
                    self.lam_region,
                    self.lam_orthog,
                    self.model.decoder_scaling,
                    self._decoder_weight_for_loss(),
                    mode=mode,
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            elif mode == "post-hoc-scaling":
                r_loss = reconstruction_loss(
                    X_reconstruction_masked,
                    truncate(X_output_masked, self.trunc),
                    r_loss_f,
                    mode="train",
                )
                r_loss = sum(
                    [r_loss[i] * v for i, v in enumerate(self.region_weights.values())]
                )
                r_loss.backward()
                self.optimizer_post_hoc.step()
                self.optimizer_post_hoc.zero_grad(set_to_none=True)
                epoch_loss_dict["reconstruction"] += r_loss.item()

            elif mode == "balance-sparsity":
                r_loss = reconstruction_loss(
                    X_reconstruction_masked,
                    truncate(X_output_masked, self.trunc),
                    r_loss_f,
                    mode="train",
                )
                r_loss = sum(
                    [r_loss[i] * v for i, v in enumerate(self.region_weights.values())]
                )
                r_loss.backward(retain_graph=True)
                r_grad = self.model._retrieve_grads()
                self.optimizer.zero_grad(set_to_none=True)

                l1_loss = torch.sum(torch.abs(Z_masked))
                l1_loss.backward(retain_graph=True)
                l1_grad = self.model._retrieve_grads()
                self.optimizer.zero_grad(set_to_none=True)

                r_grads.append(r_grad.norm(p=2))
                l1_grads.append(l1_grad.norm(p=2))

            # Accumulate losses for warmup and train modes
            if mode in ["train", "warmup"]:
                epoch_loss_dict["reconstruction"] += loss_dict["reconstruction"]
                epoch_loss_dict["latent_sparsity"] += loss_dict["latent_sparsity"]
                epoch_loss_dict["region_sparsity"] += loss_dict["region_sparsity"]
                epoch_loss_dict["orthogonality"] += loss_dict["orthogonality"]
                epoch_loss_dict["total_loss"] += loss.item()

            if mode == "evaluate":
                Z_r_masked = self.cd.mask(Z_r, truncate(Z_r_mask, self.trunc))
                [latents[k].append(Z_r_masked[k].squeeze()) for k in self.region_names]
                [
                    reconstructions[k].append(X_reconstruction_masked[k].squeeze())
                    for k in self.region_names
                ]

        # Returns are OUTSIDE the batch loop so all batches are processed
        if mode in ["train", "warmup", "evaluate", "post-hoc-scaling"]:
            return latents, reconstructions, epoch_loss_dict

        elif mode == "balance-sparsity":
            r_grads = torch.stack(r_grads)
            l1_grads = torch.stack(l1_grads)
            lam_sparse = r_grads.mean() / l1_grads.mean()
            return lam_sparse


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
        data_loader, trial_lengths = convert_to_dataloader(
            X, batch_size=1, shuffle=False
        )

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
        effective_decoder = self.model.decoder.effective_weight().cpu()
        effective_decoder_by_region = {
            k: v.cpu() for k, v in self.model.decoder.effective_weight(by_region=True).items()
        }

        checkpoint = {
            "state_dict": self.model.state_dict(),
            "effective_decoder": effective_decoder,
            "effective_decoder_by_region": effective_decoder_by_region,
            "meta": {
                "lam_sparse": self.lam_sparse,
                "lam_orthog": self.lam_orthog,
                "lam_region": self.lam_region,
                "region_weights": self.region_weights,
                "n_components": self.n_components,
                "linear": self.linear,
                "decoder_type": self.decoder_type,
                "decoder_hidden_size": self.decoder_hidden_size,
                "decoder_activation": self.decoder_activation,
            },
            "train_lambda_dicts": getattr(self, "train_lambda_dicts", None),
            "train_gradient_dicts": getattr(self, "train_gradient_dicts", None),
        }
        torch.save(checkpoint, f)

    def save_losses(self, losses: dict[str, np.ndarray], f: str):
        """
        Method for saving training losses from mSCA.fit

        Parameters
        ----------
        losses : dict[str, np.ndarray]
            Loss dictionary returned by fit().
        f : str
            Path to save losses to (.npz recommended).
        """
        np.savez(f, **losses)

    
    def load_losses(f: str) -> dict[str, np.ndarray]:
        """
        Method for loading previously saved training losses.

        Parameters
        ----------
        f : str
            Path to saved loss file (.npz).

        Returns
        -------
        dict[str, np.ndarray]s
            Dictionary of loss curves keyed by loss name.
        """
        with np.load(f, allow_pickle=False) as loaded:
            return {k: loaded[k] for k in loaded.files}

    def load(self, f: str, X: dict):
        """
        Method for loading trained model weights in mSCA

        Parameters
        ----------
        f : str
            Path to load weights from
        X : dict
            Described above
        """
        self.n_epochs = 1
        self.fit(X, load=True)


        # Load checkpoint to current device
        checkpoint = torch.load(f, map_location=self.device)

        # New format: checkpoint with metadata
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            meta = checkpoint.get("meta", {})

            # if meta:
            #     print("Checkpoint training hyperparameters:")
            #     print(f"  lam_sparse  = {meta.get('lam_sparse')}")
            #     print(f"  lam_orthog  = {meta.get('lam_orthog')}")
            #     print(f"  lam_region  = {meta.get('lam_region')}")
            #     print(f"  region_weights = {meta.get('region_weights')}")
            # else:
            #     print("Checkpoint has no saved metadata.")

        # Old format: plain state_dict only
        else:
            state_dict = checkpoint
            # print("Old checkpoint format: no saved training lam parameters found.")

        # Backward/forward compatibility for linear decoder weight naming
        if self.decoder_type.lower() == "linear":
            ckpt_param_key = "decoder.model.parametrizations.weight.original"
            ckpt_plain_key = "decoder.model.weight"

            model_keys = set(self.model.state_dict().keys())
            model_expects_param = ckpt_param_key in model_keys
            model_expects_plain = ckpt_plain_key in model_keys

            ckpt_has_param = ckpt_param_key in state_dict
            ckpt_has_plain = ckpt_plain_key in state_dict

            if model_expects_param and ckpt_has_plain and not ckpt_has_param:
                state_dict[ckpt_param_key] = state_dict.pop(ckpt_plain_key)
            elif model_expects_plain and ckpt_has_param and not ckpt_has_plain:
                state_dict[ckpt_plain_key] = state_dict.pop(ckpt_param_key)

        self.model.load_state_dict(state_dict)
    def _init_post_hoc(self):
        """
        This function intiailizes a separate optimizer to be used for
        training an un-constrained (not unit-norm) decoder during the
        last post-hoc-epoch epochs of training.
        """
        # Freeze all model weights
        for param in self.model.parameters():
            param.requires_grad = False
        

        if self.decoder_type.lower() == "linear":
            # Switch mSCA's decoder
            pre_decoder_w = self.model.decoder.model.weight.data
            pre_decoder_b = self.model.decoder.model.bias

            # Set new decoder
            new_decoder = nn.Linear(*reversed(pre_decoder_w.shape))
            new_decoder.weight.data = pre_decoder_w
            new_decoder.bias.data = pre_decoder_b
            self.model.decoder.model = new_decoder

            # Add new optimizer
            self.optimizer_post_hoc = torch.optim.Adam(
                [{"params": self.model.decoder.model.parameters(), "lr": 1e-3}]
            )
        else:
            # Keep nonlinear decoder structure and optimize its decoder layers only.(without optimize the encoder)
            self.model.decoder.hidden_layer.weight.requires_grad = True
            self.model.decoder.hidden_layer.bias.requires_grad = True
            self.model.decoder.output_layer.weight.requires_grad = True
            self.model.decoder.output_layer.bias.requires_grad = True

            self.optimizer_post_hoc = torch.optim.Adam(
                [
                    {"params": self.model.decoder.hidden_layer.parameters(), "lr": 1e-3},
                    {"params": self.model.decoder.output_layer.parameters(), "lr": 1e-3},
                ]
            )
    def _compute_grad_norms(
        self, data_loader, r_loss_f
    ) -> tuple[float, float]:
        """
        Compute mean gradient norms of reconstruction and L1 losses
        w.r.t. encoder weights across all batches. Returns (r_grad_norm, l1_grad_norm).
        Returns (0.0, 0.0) when encoder is frozen (post-hoc phase).
        """
        encoder_has_grad = any(p.requires_grad for p in self.model.encoder.parameters())
        if not encoder_has_grad:
            return 0.0, 0.0

        r_grads, l1_grads = [], []
        for _, (X_target, trial_length) in enumerate(data_loader):
            X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                self.cd.forward(X_target, trial_length)
            )
            Z, Z_r, X_reconstruction = self.model(X_input_masked)
            X_reconstruction_masked = self.cd.mask(
                X_reconstruction, truncate(output_mask, self.trunc)
            )
            Z_masked = self.cd.mask(Z, truncate(Z_mask, self.half_trunc))

            r_loss = reconstruction_loss(
                X_reconstruction_masked,
                truncate(X_output_masked, self.trunc),
                r_loss_f,
                mode="train",
            )
            r_loss = sum([r_loss[i] * v for i, v in enumerate(self.region_weights.values())])
            r_loss.backward(retain_graph=True)
            r_grad = self.model._retrieve_grads()
            self.optimizer.zero_grad(set_to_none=True)

            l1_loss = torch.sum(torch.abs(Z_masked))
            l1_loss.backward(retain_graph=True)
            l1_grad = self.model._retrieve_grads()
            self.optimizer.zero_grad(set_to_none=True)

            r_grads.append(r_grad.norm(p=2).item())
            l1_grads.append(l1_grad.norm(p=2).item())

        return float(np.mean(r_grads)), float(np.mean(l1_grads))

