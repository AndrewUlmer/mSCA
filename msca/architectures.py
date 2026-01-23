import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P

from .filters import *


class Sphere(nn.Module):
    """
    This is used to enforce a unit-norm constraint on model weights by the
    desired dimension - either rows or columns
    """

    def __init__(self, dim: int = -1):
        """
        Parameters
        ----------
        dim : int
            Axis along which to apply the unit-norm constraint.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.tensor()
            Weight matrix for a linear layer

        Returns
        ----------
        x : torch.tensor()
            input, but now unit-norm along either its rows or columns.
        """
        return x / x.norm(dim=self.dim, keepdim=True)

    def right_inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(dim=self.dim, keepdim=True)


class Encoder(nn.Module):
    """
    Encoder module for mSCA. Note that the default encoder makes use of
    tanhshrink nonlinearities which finds sparser latents than a purely
    linear model.

    Parameters
    ----------
    init_encoder : dict
        Weights computed from PCA initialization in initializations.py
    linear : bool
        Whether or not to use a linear encoder (defaults is False)
    """

    def __init__(self, init_encoder: dict, linear: bool):
        super().__init__()
        self.linear = linear

        # Storing encoder weights as separate nn.modules in dict where regions are keys
        self.model = nn.ModuleDict({})

        # Iterate through regions / weights from PCA initialization
        for region, weights in init_encoder.items():
            # If using a linear encoder, set the weights with exact PCA
            if self.linear:
                # Linear encoder
                layer = nn.Linear(weights.shape[1], weights.shape[0])

                # Initialize using PCA loadings
                layer.weight.data = torch.tensor(weights, dtype=torch.float32)
                layer.bias.data = torch.zeros(weights.shape[0], dtype=torch.float32)
                self.model[region] = layer

            # If using a nonlinear encoder, set the weights s.t. they approximate PCA
            else:
                # Note that tanhshrink is used for achieving sparser latents. Can swap out
                # for other nonlinearities if not concerned with sparsity
                self.model[region] = torch.nn.Sequential(
                    nn.Linear(weights.shape[1], weights.shape[1]),
                    nn.Tanhshrink(),
                    nn.Linear(weights.shape[1], weights.shape[0]),
                    nn.Tanhshrink(),
                    nn.Linear(weights.shape[0], weights.shape[0]),
                )

                # Use identity to get close to PCA equivalence
                self.model[region][0].weight.data = torch.eye(  # type:ignore
                    weights.shape[1]
                )

                # Initialize dimensionality reduction portion with PCA loadings
                self.model[region][2].weight.data = torch.tensor(  # type: ignore
                    weights, dtype=torch.float32
                )

                # Initialize linear part using eye-dentity
                self.model[region][4].weight.data = torch.eye(weights.shape[0])  # type: ignore

    def forward(self, X: dict) -> dict[str, torch.Tensor]:
        """
        Forward method for encoder module.

        Parameters
        ----------
        X : dict
            Keys are region names, values are tensors of shape [batch_size x time x neurons]

        Returns
        -------
        Z : dict
            Keys are region names, values are tensors of shape [batch_size x time x n_components]
        """
        return {k: self.model[k](v) for k, v in X.items()}


class Decoder(nn.Module):
    """
    Decoder module for mSCA. Note that the weights of the decoder are constrained
    to be unit-norm. This is to ensure the sparsity loss works properly.

    Parameters
    ----------
    init_decoder : dict
        Weights computed from PCA initialization in initializations.py
    init_decoder_bias : dict
        Initial bias term computed in initializations.py
    n_components : int
        The number of latent factors to use
    region_sizes : dict
        A dictionary where the keys are the names of each region and the values
        are the number of neurons in the corresponding region.
    loss_func : str
        The reconstruction loss used to train mSCA. Needed to constrain decoder
        outputs to all positive values if training on spiking data.
    """

    def __init__(
        self,
        init_decoder: dict,
        init_decoder_bias: dict,
        n_components: int,
        region_sizes: dict,
        loss_func: str,
    ):
        super().__init__()
        self.region_sizes = region_sizes
        self.loss_func = loss_func

        # Set the cumulative region sizes for setting decoder dimensionality
        self.cumsum_rs = torch.cumsum(
            torch.tensor([0] + list(self.region_sizes.values())), dim=0
        )

        # Set the output size for specifying the decoder dimensionality
        output_size = sum(region_sizes.values())

        # Create linear layer with unit-norm constraint
        self.model = P.register_parametrization(
            nn.Linear(n_components, output_size), "weight", Sphere(dim=0)
        )

        # Initialize using PCA loadings
        V_combined = torch.cat([torch.tensor(v) for v in init_decoder.values()])

        # Have to use .weight = to assign the weights, not .weight.data
        self.model.weight = torch.tensor(V_combined, dtype=torch.float32)

        # Initialize bias
        self.model.bias.data = torch.cat(
            [torch.tensor(v) for v in init_decoder_bias.values()],
            axis=1,
        ).squeeze()  # type: ignore

    def forward(self, Z: dict) -> dict[str, torch.Tensor]:
        """
        Forward method for the decoder

        Parameters
        ----------
        Z : dict
            A dictionary where the keys are region names and the values are
            tensors of shape [batch_size x time-points (truncated) x n_components]

        Returns
        -------
        X_reconstruction : dict
            A dictionary where the keys are region names and the values are tensors
            of shape [batch_size x time-points (truncated) x N_j] where N_j is the
            number of neurons in region j.
        """
        # Pre-compute region-keys
        if not hasattr(self, "region_to_idx"):
            self.region_to_idx = {k: i for i, k in enumerate(Z.keys())}

        # Iterate through region-wise latents
        X_reconstruction = {}
        for k, v in Z.items():
            # Grab appropriate indices for decoder matrix
            i = self.region_to_idx[k]
            start_idx, end_idx = self.cumsum_rs[i], self.cumsum_rs[i + 1]

            # Single matrix multiplication with bias addition
            X_reconstruction[k] = F.linear(
                v,
                self.model.weight[start_idx:end_idx],
                bias=self.model.bias[start_idx:end_idx],
            )

            # Use inverse softplus to constrain reconstructions >= 0
            if self.loss_func == "Poisson":
                X_reconstruction[k] = F.softplus(X_reconstruction[k], beta=5.0)

        return X_reconstruction

    @torch.no_grad()
    def r_loadings(self) -> dict[str, np.ndarray]:
        """
        Computes the region-specific loading from the decoder matrix to properly account for
        region-specificity learned by the model. This is used as a scalar multiplier on
        the region-specific latents for visualization purposes.

        Returns
        ----------
        rl : dict
            Keys are region names and values are tensors of L2 norm of region-specific
            portion of decoder loading for each region.
        """
        rl, d0, d1 = {}, 0, 0
        for k, r in self.region_sizes.items():
            d1 = d0 + r
            rl[k] = torch.linalg.norm(self.model.weight[d0:d1], axis=0).numpy()
            d0 = d1
        return rl


class mSCA_architecture(nn.Module):
    """
    This is a PyTorch module where the bulk of mSCA's computations are completed

    Parameters
    ----------
    init_encoder : dict
        Weights computed from PCA initialization in initializations.py
    init_decoder : dict
        Weights computed from PCA initialization in initializations.py
    init_decoder_bias : dict
        Initial bias term computed in initializations.py
    region_sizes : dict
        List of sizes for each brain region
    linear : bool
        Whether or not to use a linear encoder model (default: nonlinear)
    loss_func : str
        Reconstruction loss term used to train mSCA - used to add nonlinearity in
        Decoder if predicting Poisson rates.
    filter_length : int
        The width of the convolutional filters used to learn time-delays
    max_smoothing : int
        The maximum amount of smoothing allowable in the convolutional filters

    """

    def __init__(
        self,
        init_encoder: dict,
        init_decoder: dict,
        init_decoder_bias: dict,
        region_sizes: dict,
        n_components: int,
        linear: bool = False,
        loss_func: str = "Gaussian",
        filter_length: int = 21,
        max_smoothing: int = 10,
    ):
        super().__init__()
        self.region_sizes = region_sizes
        self.n_components = n_components
        self.linear = linear
        self.loss_func = loss_func

        # Initialize the encoder
        self.encoder = Encoder(init_encoder, self.linear)
        self.encoder_scaling = nn.Parameter(
            torch.ones(self.n_components, len(region_sizes))
        )

        # Initialize the decoder
        self.decoder = Decoder(
            init_decoder,
            init_decoder_bias,
            self.n_components,
            region_sizes,
            loss_func,
        )
        self.decoder_scaling = nn.Parameter(
            torch.ones(self.n_components, len(region_sizes))
        )

        # Initialize the convolutional filters
        self.filters = ConvolutionalFilters(
            self.n_components, len(self.region_sizes), filter_length, max_smoothing
        )

        # TESTING: add input bias
        # self.input_bias_x0 = nn.Parameter(
        #     torch.zeros(
        #         100,
        #     )
        # )
        # self.input_bias_x1 = nn.Parameter(
        #     torch.zeros(
        #         100,
        #     )
        # )

    def forward(self, X: dict) -> tuple[
        torch.Tensor,  # latent combined across regions
        dict[str, torch.Tensor],  # post-convolutional latents
        dict[str, torch.Tensor],  # reconstructions
    ]:
        """
        Forward pass throught all computations completed in mSCA

        Parameters
        ----------
        X : dict
            Keys are region names, values are tensors of shape [batch_size x time x N_j],
            where N_j is the number of neurons in region j

        Returns
        -------
        Z : torch.tensor
            This is the latent variable we apply the sparsity penalty to. It is shape
            [batch x time x n_components].
        Z_r : dict
            These are the latent variables that we visualize after fitting mSCA. It
            is a dictionary where the keys are the region names and the values are
            torch tensors of shape [batch_size x time (truncated) x n_components].
        X_reconstruction : dict
            These are the reconstructions of the neural activity input to the model.
            This is a dictionary where the keys are the region names and the values
            are torch tensors of shape [batch_size x time (truncated) x n_components]

        """
        # Mean subtraction
        X_ms = {k: v - v.mean(dim=(0, 1)) for k, v in X.items()}

        # Encode each region
        Z_r = self.encoder(X_ms)

        # Reverse filter each region's latents + smooth
        Z_r = self.filters(Z_r, mode="encode")

        # Encoder region-scaling
        Z_r = {
            k: (Z_r[k] * self.encoder_scaling[:, i]) for i, k in enumerate(Z_r.keys())
        }

        # Combine across brain regions
        Z = sum(Z_r.values())

        # Copy the latents for each brain region
        Z_r_shift = {k: Z.clone() for k, _ in Z_r.items()}  # type:ignore

        # Convolve with region-specific filters
        Z_r_shift = self.filters(Z_r_shift, mode="decode")

        # Softshrink to allow thresholding of latents for each region
        # region_scaling = F.softshrink(self.decoder_scaling)
        region_scaling = F.tanhshrink(self.decoder_scaling)

        # Clamp region scalars between -1 and 1
        region_scaling = torch.clamp(region_scaling, min=-1.0, max=1.0)

        # Apply the region-wise scaling parameter
        Z_r = {k: Z_r_shift[k] * region_scaling[:, i] for i, k in enumerate(Z_r.keys())}

        # Reconstruct the input data
        X_reconstruction = self.decoder(Z_r)

        # TESTING: 3-part decoding
        return Z, Z_r, X_reconstruction  # type: ignore
