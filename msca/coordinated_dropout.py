from typing import Union
import torch


class CoordinatedDropout:
    """
    Class used to manage coordinated dropout of inputs / outputs. This
    will generate a random checkerboard mask to apply to the inputs
    and then apply the inverted mask to the outputs.

    Parameters
    ----------
    n_components : int
        Number of latent factors being used by mSCA
    cd_rate : float (between 0 and 1.0)
        The proportion of points to dropout of the input e.g. 0.1 corresponds
        to using 90% of the input neurons and time-points to reconstruct 10%.
    filter_len : int
        The convolutional filter length used in mSCA
    """

    def __init__(
        self,
        n_components: int,
        cd_rate: float = 0.0,
        filter_len: int = 21,
        mode: str = "both",
    ):
        if cd_rate < 0.0 or cd_rate > 0.99:
            raise ValueError("CD rate must be between 0 and 1")
        self.cd_rate = cd_rate
        self.scaling_factor = 1 / (1 - self.cd_rate)
        self.n_components = n_components
        self.filter_len = filter_len
        self.mode = mode

    # @torch.compile
    def mask(
        self,
        X: Union[dict[str, torch.Tensor], torch.Tensor],
        mask: Union[dict[str, torch.Tensor], torch.Tensor],
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
        """
        Applies the binary mask used in coordinated dropout

        X : Union[dict[str, torch.Tensor], torch.Tensor]
            Input to be masked
        mask : Union[dict[str, torch.Tensor], torch.Tensor]
            A binary mask to apply to the input X (mus )
        """
        if isinstance(X, dict):
            X_masked = {k: v * mask[k] for k, v in X.items()}  # type:ignore
        else:
            X_masked = X * mask
        return X_masked

    # @torch.compile
    def mask_after_indices(self, tensor: torch.Tensor, lengths: torch.Tensor):
        """
        Sets all values after end of trial (stored in lengths) to 0.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor of shape [batch x time x neurons]
        lengths : torch.Tensor
            Tensor of shape batch - length of each trial in the batch
        """
        _, T, _ = tensor.shape
        positions = torch.arange(T).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        return tensor * mask.unsqueeze(-1)

    # @torch.compile
    def forward(self, X: dict[str, torch.Tensor], trial_length: torch.Tensor) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """
        Masks the inputs and outputs for input into mSCA. Generates
        masks for latents to truncate for trial-length

        Parameters
        ----------
        X : dict[str, torch.Tensor]
            Dictionary where the strings are region names and values
            are torch tensors of neural activity.
        trial_length : torch.Tensor
            torch.Tensor of trial lengths in the batch.
        """
        # grab first region name
        k0 = list(X.keys())[0]

        # Only generate masks non-padded portion of trial
        if self.cd_rate == 0.0:
            input_mask = {k: torch.ones_like(v) for k, v in X.items()}
            output_mask = {k: torch.ones_like(v) for k, v in X.items()}
        else:
            # Generate the input mask for each region
            if self.mode == "both":
                input_mask = {
                    k: (torch.rand_like(v) > self.cd_rate).int()
                    for _, (k, v) in enumerate(X.items())
                }
            elif self.mode == "neurons":
                input_mask = {
                    k: (torch.rand(v.shape[-1]) > self.cd_rate).int().expand(v.shape)
                    for k, v in X.items()
                }

            # Flip the input mask to get the output mask for each region
            output_mask = {k: 1 - input_mask[k] for k in X.keys()}

            # Scale up the inputs for consistency when testing without dropout
            input_mask = {k: self.scaling_factor * input_mask[k] for k in X.keys()}

        # Make mask for latents - don't want to incur sparsity for padding portion
        Z_mask = torch.ones(X[k0].shape[0], X[k0].shape[1], self.n_components)

        # Make mask for post-conv latents - need to do this for visualization
        Z_r_mask = {
            k: torch.ones(X[k0].shape[0], X[k0].shape[1], self.n_components)
            for k in X.keys()
        }

        # Set mask for padded portion of trial to 0 - I don't actually think you need this
        input_mask = {
            k: self.mask_after_indices(v, trial_length) for k, v in input_mask.items()
        }

        # Don't want filters to bleed in padding to reconstruction
        output_mask = {
            k: self.mask_after_indices(v, trial_length - self.filter_len + 1)
            for k, v in output_mask.items()
        }

        # Make mask for pre/post-convolution latents
        Z_mask = self.mask_after_indices(Z_mask, trial_length - (self.filter_len // 2))
        Z_r_mask = {
            k: self.mask_after_indices(v, trial_length - self.filter_len + 1)
            for k, v in Z_r_mask.items()
        }

        # Perform the masking
        X_input_masked = self.mask(X, input_mask)
        X_output_masked = self.mask(X, output_mask)

        return (
            X_input_masked,
            X_output_masked,
            output_mask,
            Z_mask,
            Z_r_mask,
        )  # type:ignore
