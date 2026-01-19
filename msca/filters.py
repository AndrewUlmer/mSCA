import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvolutionalFilters(nn.Module):
    def __init__(
        self,
        n_components: int,
        n_regions: int,
        filter_length: int,
        max_smoothing: int,
    ):
        """
        Initialize convolutional filters for time-shifting and smoothing latent variables.

        Parameters
        ----------
        n_components : int
            Dimensionality of the latent space.
        n_regions : int
            Number of brain regions.
        filter_length : int
            Length of the convolutional filter (must be odd).
        max_smoothing : int
            Maximum standard deviation for Gaussian smoothing (in bins, e.g., 10 ms).
        """
        super().__init__()

        # Filter configuration
        self.n_components = n_components
        self.n_regions = n_regions
        self.filter_length = filter_length
        self.max_smoothing = max_smoothing

        # Create time-points for filter evaluation
        self._filter_time = self._make_filter_time()

        # Learnable time-delays (initially 0)
        self.mus = torch.nn.Parameter(torch.zeros(self.n_components, self.n_regions))

        # Smoothing parameters to use in encoder filters - shared smoothing across regions
        self.encoder_sigmas = torch.nn.Parameter(torch.ones(self.n_components, 1) * 2.5)

        # Smoothing parameters to use in decoder filters - shared smoothing across regions
        self.decoder_sigmas = torch.nn.Parameter(torch.ones(self.n_components, 1) * 2.5)

    def _make_filter_time(self) -> torch.Tensor:
        """
        Create time-points for filter evaluation.

        Parameters
        ----------
        self : ConvolutionalFilters
            Instance of the ConvolutionalFilters class.

        Returns
        -------
        torch.Tensor
            Time points at which the filter is evaluated, ranging from
            -filter_length//2 to filter_length//2, tiled for each latent dimension.
        """
        # Generate symmetric time points around zero
        filter_time = torch.arange(
            -np.floor(self.filter_length / 2), np.ceil(self.filter_length / 2)
        )
        # Tile for each latent component
        filter_time = torch.stack([filter_time] * self.n_components, dim=0)
        return filter_time

    @torch.compile
    def _gaussian_filter(
        self, mus: torch.Tensor, sigmas: torch.Tensor, scaling: torch.Tensor, i: int
    ) -> torch.Tensor:
        """
        Create a Gaussian filter for a specific region using the provided parameters.

        Parameters
        ----------
        mus : torch.Tensor
            Time-shift parameters of shape (n_components, n_regions).
        sigmas : torch.Tensor
            Smoothing parameters of shape (n_components, n_regions).
        scaling : torch.Tensor
            Scaling parameters of shape (n_components, n_regions).
        i : int
            Index of the region for which to compute the filter.

        Returns
        -------
        torch.Tensor
            Filter evaluated at the specified time points for the given region.
        """
        return scaling[:, i][:, None] * torch.exp(
            ((self._filter_time - mus[:, i][:, None]) ** 2)
            / (-2 * sigmas[:, i][:, None] ** 2)
        )

    @torch.compile
    def _get_params(self, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get filter parameters for encoding or decoding.

        Parameters
        ----------
        mode : str
            Either "encode" or "decode". Selects which filter parameters to use.

        Returns
        -------
        mus : torch.Tensor
            Time-delay parameters (flipped for encoding).
        sigmas : torch.Tensor
            Smoothing parameters (fixed for decoding, learnable for encoding).
        """
        if mode == "encode":
            return (-self.mus, self.encoder_sigmas)
        elif mode == "decode":
            return (self.mus, self.decoder_sigmas)
        else:
            raise AssertionError("Mode must be 'encode' or 'decode'")

    @torch.compile
    def forward(
        self, Z: dict[str, torch.Tensor], mode: str = "decode"
    ) -> dict[str, torch.Tensor]:
        """
        Apply region-specific time-shifting and smoothing filters to latent variables.

        Parameters
        ----------
        Z : dict[str, torch.Tensor]
            Dictionary mapping region names to tensors of shape [batch_size, T, n_components],
            representing region-specific latents before filtering.
        mode : str, optional
            Either 'encode' or 'decode'. If 'encode', reverses the time-delays applied to each region.
            If 'decode', uses the learnable time-delays and temporal smoothing. Default is 'decode'.

        Returns
        -------
        Z_conv : dict[str, torch.Tensor]
            Dictionary mapping region names to tensors with applied region-specific smoothing and
            time-shifting.
        """
        if mode == "encode":
            mus, sigmas = self._get_params(mode)

            # Not allowing smoothing in the encoder
            sigmas = torch.clamp(sigmas, min=1, max=10)

            # Copy smoothing params across regions
            sigmas = torch.cat([sigmas] * mus.shape[1], dim=1)

            # Make scaling the same for all regions
            scaling = torch.ones_like(sigmas)

        elif mode == "decode":
            # These are the actualy learnable time-delays
            mus, sigmas = self._get_params(mode)

            # Preventing the model from shrinking the standard deviations
            sigmas = torch.clamp(sigmas, min=1, max=10)

            # Copy smoothing params across regions
            sigmas = torch.cat([sigmas] * mus.shape[1], dim=1)

            # Make scaling the same for all regions
            scaling = torch.ones_like(sigmas)

        # Clamping the time-delays
        mus = torch.clamp(mus, min=-10, max=10)

        # Iterate through regions and filter the latents with respect to time
        Z_conv = {}
        for i, (region, z) in enumerate(Z.items()):
            # Grab region-specific filter kernel
            filter_kernel = self._gaussian_filter(mus, sigmas, scaling, i)

            # Re-normalize filter to force integration to 1
            filter_kernel = (
                filter_kernel / filter_kernel.sum(axis=1)[:, None]  # type:ignore
            )

            # Perform convolution
            z_filtered = F.conv1d(
                z.permute(0, 2, 1), filter_kernel[:, None], groups=z.shape[2]
            ).permute(0, 2, 1)

            # Add the filtered result to output dict
            Z_conv[region] = z_filtered

        return Z_conv
