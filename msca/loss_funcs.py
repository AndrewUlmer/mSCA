import numpy as np
from typing import Union, Callable

import torch
import torch.nn.functional as F
from torch.nn.functional import poisson_nll_loss


def gaussian_f(
    X_reconstruction: Union[torch.Tensor, np.ndarray],
    X: Union[torch.Tensor, np.ndarray],
    mode: str = "train",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Wrapper function that evaluates reconstructions against X using the Gaussian
    SSE loss

    Arguments
    ----------
    X_reconstruction : torch.tensor or np.array
        Reconstructed neural activity
    X : torch.tensor or np.array
        Grount-truth neural activity

    Returns
    -------
    loss : torch.float32
        Gaussian loss
    """
    # Check if input is a numpy array
    is_numpy = isinstance(X_reconstruction, np.ndarray)

    # Convert to torch tensors if not already
    if is_numpy:
        X = torch.tensor(X)
        X_reconstruction = torch.tensor(X_reconstruction)

    # Sum the loss over neurons and time-points when training / initializing
    if mode == "train":
        loss = (((X_reconstruction - X)) ** 2).sum()

    # Compute the elementwise loss for bootstrapping if evaluating
    elif mode == "evaluate":
        loss = ((X_reconstruction - X)) ** 2

    # Convert back to numpy array if needed
    return loss.numpy() if is_numpy else loss


def poisson_f(
    X_reconstruction: Union[torch.Tensor, np.ndarray],
    X: Union[torch.Tensor, np.ndarray],
    mode: str = "train",
) -> Union[torch.Tensor, np.ndarray]:
    """
    Wrapper function that evaluates reconstructions against X using the Poisson
    negative log-likelihood

    Arguments
    ----------
    X_reconstruction : torch.tensor or np.array
        Reconstructed neural activity
    X : torch.tensor or np.array
        Grount-truth neural activity

    Returns
    -------
    loss : torch.float32
        Poisson loss
    """
    # Check if input is a numpy array
    is_numpy = isinstance(X_reconstruction, np.ndarray)

    # Convert to torch tensors if not already
    if is_numpy:
        X = torch.tensor(X)
        X_reconstruction = torch.tensor(X_reconstruction)

    # Sum the loss over neurons and time-points when training / initializing
    if mode == "train":
        loss = poisson_nll_loss(
            X_reconstruction, X, log_input=False, reduction="sum"  # type: ignore
        )
    # Compute the elementwise loss for bootstrapping if evaluating
    elif mode == "evaluate":
        loss = poisson_nll_loss(
            X_reconstruction, X, log_input=False, reduction="none"  # type: ignore
        )

    # Convert back to numpy array if needed
    return loss.numpy() if is_numpy else loss


def reconstruction_loss(
    X_inp: dict, X_tgt: dict, eval_func: Callable, mode: Union[str, None] = None
) -> list[torch.Tensor]:
    """
    Computes the reconstruction loss of the PCA
    reconstructions using the specified evaluation fn.

    Parameters
    ----------
    X_inp : dict
        Dict of reconstructed neural activity, concatenated
        across trials - keys are region names
    X_tgt : dict
        Dict of ground-truth neural activity, concatenated
        across trials - keys are region names
    eval_func : function
        Function used to compute reconstruction performance.
        Defined in utils.

    Returns
    -------
    recon_loss : list
        Each entry is the reconstruction loss for the
        corresponding region.

    """
    return [
        eval_func(x_i, x_h, mode=mode)
        for x_i, x_h in zip(X_inp.values(), X_tgt.values())
    ]


def region_sparsity_loss(z: torch.Tensor, scaling: torch.Tensor) -> torch.Tensor:
    """
    Computes the region-sparsity loss. Weights the penalty on the region-scaling
    parameter by the magnitude of the latent to focus gradients on signals
    that are meaningful for reconstruction.

    Parameters
    ----------
    z : torch.Tensor
        The latent after combining across regions (see mSCA_architecture.forward())
    scaling : torch.Tensor
        A region-specific scalar parameter used to encourage region-specific solutions
        in the latents signals.
    """
    # Set the magnitude function - using lambda in case we decide to change
    mag_f = lambda x: x.std(dim=(0, 1))

    # Detaching magnitude calculation so this loss function doesn't affect latent shape
    z_mag = mag_f(z).detach()

    # Computing region-sparsity loss weighting scalars by latent magnitude
    L = (scaling * z_mag[:, None]).abs().sum(axis=1).sum()
    return L


def gaussian_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rws: dict[str, float],
    z: torch.Tensor,
    lam_sparse: float,
    lam_region: float,
    lam_orthog: float,
    scaling: torch.Tensor,
    V: torch.Tensor,
    mode: Union[str, None] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Arguments
    ----------
    input : dict
        Keys are region names, values are torch tensors
        containing reconstructed neural activity.
    target : dict
        Keys are region names, values are torch tensors
        containing the neural activity we're trying to
        reconstruct.
    rws : dict
        Keys are region names, values are the weights
        to apply to each region's reconstruction loss.
    z : torch.tensor
        Latent prior to smoothing / time-shifting which
        is where we will apply the L1 sparsity penalty.
    lam_sparse: float
        Weight to apply to sparsity penalty - the
        higher this is, the sparser the latents will be.
    V : torch.tensor
        Decoder matrix used to reconstruct the neural
        activity. We wil apply group-sparsity to this.

    Returns
    ----------
    loss : torch.tensor
        Returns the weighted sum of the Gaussian
        reconstruction loss, L1 sparsity loss, and the
        region-sparsity loss.
    """

    # Compute Gaussian SSE loss
    rc = reconstruction_loss(input, target, gaussian_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    # Compute sparsity loss
    l1 = torch.sum(torch.abs(z))

    # Compute the group-sparsity loss
    gs = region_sparsity_loss(z, F.softshrink(scaling))

    # Compute the orthogonality loss
    orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2

    return (
        rc + l1 * lam_sparse + gs * lam_region + lam_orthog * orth,
        {
            "reconstruction": rc.item(),  # type: ignore
            "latent_sparsity": l1.item(),
            "region_sparsity": gs.item(),
            "orthogonality": orth.item(),
        },
    )


def poisson_loss(
    input: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    rws: dict[str, float],
    z: torch.Tensor,
    lam_sparse: float,
    lam_region: float,
    lam_orthog: float,
    scaling: torch.Tensor,
    V: torch.Tensor,
    mode: Union[str, None] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Arguments
    ----------
    input : dict
        Keys are region names, values are torch tensors
        containing reconstructed neural activity.
    target : dict
        Keys are region names, values are torch tensors
        containing the neural activity we're trying to
        reconstruct.
    rws : dict
        Keys are region names, values are the weights
        to apply to each region's reconstruction loss.
    z : torch.tensor
        Latent prior to smoothing / time-shifting which
        is where we will apply the L1 sparsity penalty.
    lam_sparse: float
        Weight to apply to sparsity penalty - the
        higher this is, the sparser the latents will be.
    V : torch.tensor
        Decoder matrix used to reconstruct the neural
        activity. We wil apply group-sparsity to this.

    Returns
    ----------
    loss : torch.tensor
        Returns the weighted sum of the Poisson
        reconstruction loss, L1 sparsity loss, and the
        region-sparrsity loss.
    """

    # Compute Poisson reconstruction loss
    rc = reconstruction_loss(input, target, poisson_f, mode)

    # Weight each region separately
    rc = sum([x * rw for x, rw in zip(rc, rws.values())])

    # Compute sparsity loss
    l1 = torch.sum(torch.abs(z))

    # Compute the group-sparsity loss
    # gs = region_sparsity_loss(z, F.softshrink(scaling))
    gs = region_sparsity_loss(z, F.tanhshrink(scaling))

    # Compute the orthogonality loss
    orth = torch.norm(V.T @ V - torch.eye(V.shape[1], device=V.device)) ** 2

    return (
        rc + l1 * lam_sparse + gs * lam_region + lam_orthog * orth,
        {
            "reconstruction": rc.item(),  # type: ignore
            "latent_sparsity": l1.item(),
            "region_sparsity": gs.item(),
            "orthogonality": orth.item(),
        },
    )
