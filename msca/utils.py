import torch
import numpy as np
import scipy as sp
from collections import namedtuple


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """
    Performs Gaussian smoothing of time-series.

    Arguments
    ----------
    x : np.ndarray
        Time x N_i array of neural activity for region i
    s : float
        Standard deviation to use in Gaussian.

    Returns
    -------
    x_smoothed : np.ndarray
        Neurons x Time array of smoothed neural
        activity.
    """

    return sp.ndimage.gaussian_filter1d(x, sigma=sigma, axis=0, mode="constant")


def presmooth(X: dict, sigma: float = 5.0) -> dict:
    """
    Function that smooths neural activity with respect
    to time, using a Gaussian filter.

    Arguments
    ----------
    X : dict
        Format described in quickstart.ipynb
    sigma : float
        Standard deviation to use in Gaussian. Units
        are in bins.

    Returns
    -------
    X_smoothed : dict
        Same format as X, but now smoothed.
    """
    # Iterate through regions / trials and smooth
    X_smooth = {}
    for r_name, x_r in X.items():
        X_smooth[r_name] = [
            gaussian_smooth(x_r_i.astype("float32"), sigma) for x_r_i in x_r
        ]
    return X_smooth


def region_sizes(X_dict: dict, cumulative: bool = True) -> list:
    """
    Returns an list of the number of neurons in each region

    Parameters
    ----------
    X_dict : dict
        Format described in quickstart.ipynb
    cumulative : bool
        Whether or not to compute the cumulative number of neurons
    """
    region_sizes = np.cumsum([0] + [v.shape[1] for _, v in X_dict.items()])
    if cumulative:
        return region_sizes.tolist()
    else:
        return np.diff(region_sizes).tolist()


def split_into_regions(X_concat: np.ndarray, X_dict: dict) -> dict:
    """
    Splits an array of size (T x (N_1 + N_2 + ... + N_i)) into region dictionary
    described in quickstart.ipynb

    Parameters
    ----------
    X_concat : np.ndarray
        Concatenated neural data across all regions (time x total_neurons).
    X_dict : dict
        Dictionary mapping region names to original neural data arrays (used for region sizes).

    Returns
    -------
    X_out : dict
        Dictionary mapping region names to their corresponding slices of X_dict.
    """
    X_out = {}
    sizes = region_sizes(X_dict)
    for region, start, end in zip(X_dict.keys(), sizes[:-1], sizes[1:]):
        X_out[region] = X_concat[:, start:end]
    return X_out


def dict_of_lists_torchify(X_dict):
    """
    Arguments
    ----------
    X_dict : dict
        Described in ./msca/models.py:fit()

    Returns
    -------
    X_dict : dict
        Same as inputs, but now the trials
        are torch tensors instead of numpy
        arrays.
    """
    return {
        k: [torch.tensor(x, dtype=torch.float32) for x in v] for k, v in X_dict.items()
    }


def trim(X, filter_len, num_convs=1):
    """
    Arguments
    ----------
    X: dict
        Keys are region names and values are torch tensors
        of shape [batch x time x neurons]
    filt_len : int
        The length of the filter used in the convolutions
        in mSCA.

    Returns
    -------
    X_t: dict
        Same as inputs, but now truncated on either side
        to account for convolving with no padding
    """
    trunc = int(np.floor(filter_len / 2)) * num_convs
    return {k: v[:, trunc:-trunc] for k, v in X.items()}


def mask(Z, X_inp, X_tgt, M_c, M_f, filter_len):
    """
    This function handles the masking of the latents,
    reconstructions, and targets.

    Arguments
    ----------
    Z : torch.tensor [batch x time x latents]
        We don't want to apply the sparsity penalty
        to portions of the trial that we padded with
        zeros.
    X_inp : dict
        This contains the predicted reconstructions. The
        keys are region names and values are torch.tensors().
        We don't want to reconstruct the portions of the
        input that have been padded with zeros.
    X_tgt : dict
        This contains the targets. The keys are region names
        and values are torch.tensors(). We have to trim this
        to account for not using padding in the convolutions.
    M : dict
        Contains tensors for appropriate masking.

    Returns
    -------
    Z : torch.tensor [batch x time x latents]
        Original Z, but with latents zeroed out
        in the padded portions.
    X_inp : dict
        Origional X_inp, but with reconstructions zeroed out
        at the padded portions.
    X_tgt : dict
        Original X_tgt, but truncated to match the result
        post-conv x2

    """
    # Trim the targets according to the convolution length
    X_tgt_t = trim(X_tgt, filter_len, num_convs=2)

    # Trim the masks to account for the conv length for recon
    M_t_i = trim(M_c, filter_len, num_convs=2)

    # Trim the masks to account for the conv length for latent
    M_t_z = trim(M_f, filter_len)

    # Multiply the mask with the reconstruction
    X_inp_t = {k: v * M_t_i[k] for k, v in X_inp.items()}

    # Multiply the mask with the latent

    # THIS IS WRONG - NEED TO FILL IN M_t_z
    k = list(M_t_z.keys())[0]
    Z_t = M_t_z[k][:, :, : Z.shape[-1]] * Z

    return Z_t, X_inp_t, X_tgt_t


def detect_deflect(X, trial_num=0):
    """
    Sorts signals by peak deflection time

    Arguments
    ----------
    X : dict
        Keys are region names, values are lists of trials
        where each trial is a T x K np.array(). Each trial
        should have K cols, where K is the number of latents.

    Returns
    ----------
    idxs : np.array
        Inputs sorted by time of peak deflection from the
        mean.
    """
    # Sum the latents across brain regions
    x_c = sum([v[trial_num] for _, v in X.items()])

    # Grab the number of latent dimensions
    K = x_c.shape[-1]  # type:ignore

    # Compute the deviations of summed latents from mean
    devs = (x_c - x_c.mean(axis=0)) ** 2  # type:ignore

    # Find where deviations from mean start
    idxs = [
        np.where((devs > x_c.std(axis=0))[:, i])[0][0] for i in range(K)  # type:ignore
    ]

    # Sort the dimensions according to their first deflections
    idxs = np.array(idxs).argsort()
    return idxs


def to_list_of_dicts(X):
    """
    Arguments
    ----------
    X : dict
        Dictionary where keys are region names and values are tensors where
        the first dimension is the number of trials in the dataset

    Returns
    -------
    X : list of dicts
        Same as inputs, but now the trials are lists of dicts
    """
    n_trials = len(next(iter(X.values())))
    return [{region: X[region][i] for region in X} for i in range(n_trials)]


def truncate(X, trunc):
    """
    Arguments
    ----------
    X : dict
        Described in ./msca/models.py:fit()
    trunc : slice
        The slice to truncate the data to

    Returns
    -------
    X : dict
        Same as inputs, but now truncated
    """
    if isinstance(X, dict):
        return {k: X[k][:, trunc] for k in X.keys()}
    else:
        return X[:, trunc]


def torchify(X):
    """
    Parameters
    ----------
    X : dict
        Format described in quickstart.ipynb

    Returns
    -------
    output : dict
        The same format as X, but now each np.ndarray is a torch.tensor
    """
    return {k: [torch.tensor(x, dtype=torch.float32) for x in v] for k, v in X.items()}


def pad_trials(X_tensor):
    """
    Utility function for padding all trials to be the same length as the
    longest trial -- needed for batching during training

    Parameters
    ----------
    X_tensor : dict
        Dictionary with format described in quickstary.ipynb - values are lists
        of torch.tensors instead of np.ndarrays

    Returns
    -------
    X_padded : dict
        Values are now torch.tensors where the first dimension is the total number
        of trials in the dataset
    lengths : list
        A list of all the pre-padding trial lengths in the dataset
    """

    # Retrieve the lengths of each trial
    k0 = list(X_tensor.keys())[0]
    lengths = [x.shape[0] for x in X_tensor[k0]]

    # Pad the trials to be the same length
    X_padded = {
        k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True)
        for k, v in X_tensor.items()
    }

    return X_padded, lengths


def to_named_tuples(X_list_of_dicts, trial_lengths):
    """
    Converts list of dictionaries to list of named tuples.
    This is used for batching during training

    Parameters
    ----------
    X_list_of_dicts: list
        Each entry is a dictionary where the keys are the region names and the values
        are the neural activity for each region on that trial

    Returns
    -------
    output : list
        A list of named tuples including the data and the trial length for each
        corresponding trial. This is used for masking during training.
    """

    Trial = namedtuple("Trial", ["X", "trial_length"])
    return [
        Trial(X, trial_length)
        for X, trial_length in zip(X_list_of_dicts, trial_lengths)
    ]


def inv_softplus(x: np.ndarray, beta=5.0) -> np.ndarray:
    """
    Inverse softplus function used to initialize the decoder biases.
    """
    return (1 / beta) * np.log(np.exp(beta * x) - 1)
