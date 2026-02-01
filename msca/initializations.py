from typing import Callable
from sklearn.decomposition import PCA

from .utils import *
from .loss_funcs import *

from msca.evaluations import PoissonRegressorWrapper


def _pca_reconstruction(
    X: dict, n_components: int, loss_func: str, X_orig: dict  #### TESTING POISSON GLM
) -> tuple[np.ndarray, dict, dict, dict, dict]:
    """
    Computes the reconstruction of the neural activity using principal component analysis

    Parameters
    ----------
    X : dict
        Format described in quickstart.ipynb, but now all trials are concatenated into a np.ndarray
        for each region
    n_components : int
        The number of latent factors used in PCA
    loss_func : str
        The loss function - need to make reconstructions >= 0 for spiking data

    Returns
    -------
    Z : np.ndarray
        latents computed using PCA
    U : dict
        PCA loadings to be used as mSCA's initial encoder
    V : dict
        PCA loadings to be used as mSCA's initial decoder
    X_reconstruction : dict
        Reconstruction from using PCA
    b_enc_init : np.ndarray
        Initial bias term to use in PCA's encoder
    b_dec_init : dict
        Initial bias term to use in PCA's decoder
    """
    # Concatenate X across regions
    X_r_concat = np.concatenate(list(X.values()), axis=1)

    # Fit PCA
    pca = PCA(n_components=n_components).fit(X_r_concat)

    # Compute latents
    Z = pca.transform(X_r_concat)

    # Compute reconstructions
    X_reconstruction = pca.inverse_transform(Z)

    # Use ReLU to constrain reconstructions >= 0
    if loss_func == "Poisson":
        X_reconstruction = np.maximum(X_reconstruction, 0)

    #### TESTING: USE POISSON GLM INSTEAD OF PCA + RELU

    # concatenate X_orig
    # x_orig_cat = np.concatenate([v for v in X_orig.values()], axis=1)
    # glm = PoissonRegressorWrapper(alpha=0.001).fit(Z, x_orig_cat)
    # X_reconstruction = glm.predict(Z)

    # Partitions back into regions
    X_reconstruction = split_into_regions(X_reconstruction, X)

    # Split U and V into region dicts
    U = split_into_regions(pca.components_, X)
    V = {k: u.T for k, u in U.items()}

    # TODO: does it make more sense to have this before the encoding?
    if loss_func == "Poisson":
        b_dec_init = inv_softplus(X_r_concat.mean(axis=0))
    else:
        b_dec_init = X_r_concat.mean(axis=0)

    # Split for multiple brain regions
    b_dec_init = split_into_regions(b_dec_init.reshape(1, -1), X)

    # Return results
    return Z, U, V, X_reconstruction, b_dec_init


def _compute_relative_reconstruction_loss(
    X_concat: dict, X_reconstruction: dict, eval_func: Callable
) -> dict:
    """
    Parameters
    ----------
    X_concat : dict
        Format described in quickstart.ipynb, but now all trials are concatenated into a np.ndarray
        for each region
    X_reconstruction : dict
        Same as X_concat, except for reconstructions, not the original neural activity

    Returns
    -------
    l_rel : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted off.
    """
    # Concatenate ground-truth data across regions
    X_r_concat = np.concatenate(list(X_concat.values()), axis=1)

    # Concatenate PCA + presmoothing reconstruction across regions
    X_r_reconstruction = np.concatenate(list(X_reconstruction.values()), axis=1)

    # Compute the reconstruction loss using "perfect" reconstructions
    l_perf = eval_func(X_r_concat, X_r_concat, mode="evaluate")
    l_perf = {k: v.sum() for k, v in split_into_regions(l_perf, X_concat).items()}

    #### TESTING: NULL MODEL
    null = np.stack([X_r_concat.mean(axis=0)] * X_r_concat.shape[0])
    l_null = eval_func(null, X_r_concat, mode="evaluate")
    l_null = {k: v.sum() for k, v in split_into_regions(l_null, X_concat).items()}

    # Compute the reconstruction loss using the real reconstructions
    l_real = eval_func(X_r_reconstruction, X_r_concat, mode="evaluate")
    l_real = {k: v.sum() for k, v in split_into_regions(l_real, X_concat).items()}

    # Compute the relative reconstruction loss
    l_rel = {k: (l_real[k] - l_perf[k]) for k in l_real.keys()}

    #### TESTING: NULL MODEL
    l_rel = {k: np.abs(l_real[k] - l_null[k]) for k in l_real.keys()}

    return l_rel


def _compute_region_weights(relative_reconstruction_loss: dict) -> dict:
    """
    Computes weights for balancing the reconstruction loss across brain regions. More specifically,
    it weights each regions' reconstruction loss such that it's equal to the mean reconstruction
    loss across regions.

    Parameters
    ----------
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted off.

    Returns
    -------
    region_weights : dict
        A dictionary where the weights are to be applied to the reconstruction loss for each region
        so that mSCA doesn't preferentially find latents more representative of one region over
        another.
    """
    # Compute egion weights to balance loss across regions
    mean_rel_error = sum(relative_reconstruction_loss.values()) / len(
        relative_reconstruction_loss.values()
    )
    return {
        k: (mean_rel_error / v).item() for k, v in relative_reconstruction_loss.items()
    }


def _compute_lam_sparse(
    Z: np.ndarray,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the sparsity penalty (lam_sparse) as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        Latent representations computed using PCA (samples x n_components).
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default sparsity level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Sparsity penalty (lam_sparse) computed relative to the reconstruction loss.
    """

    # Compute the L1 norm of the latents
    L_sparse = np.abs(Z).sum()

    # If the user has not manually passed a sparsity value to use
    if pct is None:
        pct = 0.1 if loss_func == "Gaussian" else 0.05

    # Make lambda sparse such that L1 is pct% of reconstruction
    return sum(relative_reconstruction_loss.values()) * pct / L_sparse


def _compute_lam_orthog(
    n_components: int,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial orthogonality loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Orthogonality penalty (lam_orthog) computed relative to the reconstruction loss.
    """
    # Compute "initial" orthogonality loss estimate
    L_orth = np.sum(n_components * (n_components - 1) * 0.01)

    # If the user has not manually passed a orthogonality value to use defaults
    if pct is None:
        pct = 0.1 if loss_func == "Gaussian" else 0.01

    # Make lambda sparse such that L1 is pct% of reconstruction
    return sum(relative_reconstruction_loss.values()) * pct / L_orth


def _compute_lam_region(
    Z: np.ndarray,
    n_components: int,
    relative_reconstruction_loss: dict,
    loss_func: str,
    pct: Union[None, float] = None,
) -> float:
    """
    Compute the orthogonality penalty as a fraction of the reconstruction loss.

    Parameters
    ----------
    Z : np.ndarray
        The latents inferred using PCA
    n_components: dict
        The number of latent factors used to fit mSCA / PCA
    relative_reconstruction_loss : dict
        Dictionary where the keys are region names and the values are the real reconstruction loss
        with the perfect reconstruction loss subtracted.
    loss_func : str
        Loss function string - used to set the default orthogonality level.
    pct : [float, None], optional
        Fraction of the reconstruction loss to set as the initial region sparsity loss. If None, it will
        use the defaults.

    Returns
    -------
    float
        Region sparsity penalty (lam_region) computed relative to the reconstruction loss.
    """
    # Set the magnitude function - using lambda function in case we change it
    mag_f = lambda x: np.std(x, axis=0)

    # Compute the magnitude across all latents
    mag_z = mag_f(Z)

    # Penalty is applied to tensor of (n_components x n_regions)
    n_regions = len(relative_reconstruction_loss.keys())

    # Estimate region-scaling parameters - assuming 10% of the latents are region-specific
    C = np.random.binomial(n=1, p=0.9, size=(n_components, n_regions))

    # Find total contribution - want sparsity across regions (each row)
    L_region = np.abs(C * mag_z[:, None]).sum(axis=1).sum()

    # If the user has not manually passed a region-sparsity value use defaults
    if pct is None:
        # TODO: determine if we want this to be defaulted or nah
        pct = 0.1 if loss_func == "Gaussian" else 0.01
        # pct = 0.0

    # Compute the region-sparsity penalty
    return sum(relative_reconstruction_loss.values()) * pct / L_region


def _initialize(
    X: dict,
    n_components: int,
    loss_func: str,
    lam_sparse: Union[None, float],
    lam_orthog: Union[None, float],
    lam_region: Union[None, float],
) -> tuple[
    dict,  # encoder
    dict,  # decoder
    dict,  # decoder bias
    float,  # lam_sparse
    float,  # lam_orthog
    float,  # lam_region
    dict,  # region weights
    dict,  # region sizes
]:
    # Check for valid loss function
    if loss_func not in ["Gaussian", "Poisson"]:
        raise NotImplementedError

    # Pre-smooth for PCA if using spiking data
    X_smoothed = presmooth(X) if loss_func == "Poisson" else X  # type: ignore

    # Concatenate non-smoothed data to use as target in reconstruction loss
    X_concat = {region: np.concatenate(trials) for region, trials in X.items()}

    # Concatenate smoothed data for computing PCA reconstruction
    X_smoothed_concat = {
        region: np.concatenate(trials) for region, trials in X_smoothed.items()
    }

    # Compute the PCA reconstruction
    Z, U, V, X_reconstruction, b_dec_init = _pca_reconstruction(
        X_smoothed_concat,
        n_components=n_components,
        loss_func=loss_func,
        X_orig=X_concat,  ### TESTING POISSON GLM
    )

    # Compute the relative reconstruction loss
    eval_func_name = f"{loss_func.lower()}_f"
    eval_func = eval(eval_func_name)
    relative_reconstruction_loss = _compute_relative_reconstruction_loss(
        X_concat, X_reconstruction, eval_func
    )

    # Balance the reconstruction loss across regions
    rws = _compute_region_weights(relative_reconstruction_loss)

    # Compute the latent sparsity loss as a function of initial reconstruction loss
    lam_sparse = _compute_lam_sparse(
        Z, relative_reconstruction_loss, loss_func, pct=lam_sparse
    )

    # Compute the orthogonality loss as a function of the initial reconstruction loss
    lam_orthog = _compute_lam_orthog(
        n_components, relative_reconstruction_loss, loss_func, pct=lam_orthog
    )

    # Compute the region sparsity loss as a function of initial reconstruction loss
    lam_region = _compute_lam_region(
        Z, n_components, relative_reconstruction_loss, loss_func, pct=lam_region
    )

    # Compute the region sizes and convert to dictionary
    rs = region_sizes(X_concat, cumulative=False)
    rs = {k: rs[i] for i, k in enumerate(rws.keys())}

    return U, V, b_dec_init, lam_sparse, lam_orthog, lam_region, rws, rs
