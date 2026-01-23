import numpy as np


def _create_sine(T_sine: int, c_sine: int) -> np.ndarray:
    """
    Creates a windowed sine wave.
    """
    tau = T_sine / (2 * np.pi) / c_sine
    return np.sin(np.arange(0, T_sine) / tau)


def _create_filter(shift: int, max_shift: int) -> np.ndarray:
    """
    Creates a delta function used to shift the ground-truth latent
    representation in time for each region.
    """
    f = np.zeros(max_shift * 2 + 1)
    f[max_shift + shift] = 1
    return f


def simulate_trial_averages(
    T: int = 600,  # Number of timepoints per trial
    n_trials: int = 10,  # Number of trials (trial averages)
    n_regions: int = 2,  # Number of brain regions
    n_neurons: int = 100,  # Number of neurons per region
    max_shift: int = 5,  # Maximum time shift (in either direction)
    r_sim: int = 5,  # Number of simulated latent dimensions
    noise_level: float = 0.1,  # Noise level for trials
    random_seed: int = 1,
) -> tuple[dict, dict, np.ndarray]:
    """
    Simulates a trial-average dataset for mSCA with specified parameters.

    Parameters
    ----------
    T : int
        Number of timepoints per trial.
    n_trials : int
        Number of trials (trial averages).
    n_regions : int
        Number of brain regions.
    n_neurons : int
        Number of neurons per region.
    max_shift : int
        Maximum time shift (in either direction).
    r_sim : int
        Number of simulated latent dimensions.
    noise_level : float
        Standard deviation of the Gaussian noise added to trials.
    random_seed : int
        Seed for numpy random number generator for reproducibility.

    Returns
    -------
    X : dict
        A dictionary containing the simulated data in the format
        {'Region_0': [trial_1, ...], 'Region_1': [trial_1, ...]}.
    Z_gt : np.ndarray
        Ground-truth latent factors.
    delays_gt : np.ndarray
        Ground-truth delays for each latent factor.

    """
    np.random.seed(random_seed)

    if n_regions > 2:
        # TODO: allow for simulating n_regions - currently only supports 2
        raise NotImplementedError

    # Matrices that project low dimensional space into high-D "neural space"
    V0 = np.random.randn(n_neurons, r_sim)
    V1 = np.random.randn(n_neurons, r_sim)

    # Make projections unit-norm
    V0 = V0 / np.linalg.norm(V0, axis=0)
    V1 = V1 / np.linalg.norm(V1, axis=0)

    # Create ground-truth (simulated) latents
    Z = np.zeros([T + 2 * max_shift, r_sim])
    for i in range(r_sim):
        Z[max_shift + (100 * i) : max_shift + (100 * i) + 200, i] = _create_sine(
            200, i + 1
        )

    # Create ground-truth time-shifts for dims 1-3
    delays = np.zeros(r_sim, dtype="int32")
    if r_sim > 1:
        delays[1] = np.random.randint(low=-max_shift, high=max_shift)
    if r_sim > 3:
        delays[3] = np.random.randint(low=-max_shift, high=max_shift)

    ## TESTING: 1-bin delay
    delays[3] = 1

    # Time-shift latents
    Z0, Z1 = np.zeros_like(Z), np.zeros_like(Z)
    f0s, f1s = [], []
    for dim, delay in enumerate(delays):
        f0 = _create_filter(delay, max_shift=max_shift)
        f1 = _create_filter(-delay, max_shift=max_shift)

        if dim == 3:
            f1 = _create_filter(0, max_shift=max_shift)

        Z0[:, dim] = np.convolve(Z[:, dim], f0, mode="same")
        Z1[:, dim] = np.convolve(Z[:, dim], f1, mode="same")

        f0s.append(f0), f1s.append(f1)  # type: ignore

    # Compute the ground-truth delays for comparison to trained model
    delays_gt = [(f0_i.argmax() - f1_i.argmax()).item() for f0_i, f1_i in zip(f0s, f1s)]

    # Make region-specific dimensions
    Z0_final = Z0.copy()
    Z1_final = Z1.copy()

    if r_sim > 0:
        Z0_final[:, -1] = 0
    if r_sim > 0:
        Z1_final[:, 0] = 0

    # Pad the latents to account for mSCA truncation
    padding = np.zeros((max_shift * 2 + 10, r_sim))
    Z0_padded = np.concatenate((padding, Z0_final, padding))
    Z1_padded = np.concatenate((padding, Z1_final, padding))

    # Project latents up into firing-rate space
    X0_fr, X1_fr = Z0_padded @ V0.T, Z1_padded @ V1.T

    # Create trials by adding noise to high-D latents
    X0_trials, X1_trials = [], []
    for tri_num in range(n_trials):
        X0_trials.append((X0_fr + noise_level * np.random.randn(*X0_fr.shape)))
        X1_trials.append((X1_fr + noise_level * np.random.randn(*X1_fr.shape)))

    # Add noised neural activity to a dictionary
    X = {}
    X["X0"] = X0_trials
    X["X1"] = X1_trials

    # Add ground-truth latents to a dictionary
    Z_gt = {}
    Z_gt["Z0"] = Z0_final
    Z_gt["Z1"] = Z1_final

    return X, Z_gt, delays_gt  # type: ignore


def simulate_single_trials(
    T: int = 600,  # Number of timepoints per trial
    n_trials: int = 50,  # Number of trials (trial averages)
    n_regions: int = 2,  # Number of brain regions
    n_neurons: int = 150,  # Number of neurons per region
    max_shift: int = 5,  # Maximum time shift (in either direction)
    r_sim: int = 5,  # Number of simulated latent dimensions
    avg_fr: int = 30,  # Average firing rate to use for simulation
    dts: float = 1e-2,  # Simulated bin size
    random_seed: int = 1,  # Random seed to use
) -> tuple[dict, dict, np.ndarray]:
    """
    Simulates single-trials - most of this code is redundant with respect
    to simulate trial-averages, but I'm leaving it in for didactic purposes.

    Parameters
    ----------
    T : int
        Number of timepoints per trial.
    n_trials : int
        Number of trials (trial averages).
    n_regions : int
        Number of brain regions.
    n_neurons : int
        Number of neurons per region.
    max_shift : int
        Maximum time shift (in either direction).
    r_sim : int
        Number of simulated latent dimensions.
    random_seed : int
        Seed for numpy random number generator for reproducibility.

    Returns
    -------
    X : dict
        A dictionary containing the simulated data in the format
        {'Region_0': [trial_1, ...], 'Region_1': [trial_1, ...]}.
    Z_gt : np.ndarray
        Ground-truth latent factors.
    delays_gt : np.ndarray
        Ground-truth delays for each latent factor.

    """
    np.random.seed(random_seed)

    if n_regions > 2:
        # TODO: allow for simulating n_regions - currently only supports 2
        raise NotImplementedError

    # Matrices that project low dimensional space into high-D "neural space"
    V0 = np.random.randn(n_neurons, r_sim)
    V1 = np.random.randn(n_neurons, r_sim)

    # Make projections unit-norm
    V0 = V0 / np.linalg.norm(V0, axis=0)
    V1 = V1 / np.linalg.norm(V1, axis=0)

    # Create ground-truth (simulated) latents
    Z = np.zeros([T + 2 * max_shift, r_sim])
    for i in range(r_sim):
        Z[max_shift + (100 * i) : max_shift + (100 * i) + 200, i] = _create_sine(
            200, i + 1
        )

    # Create ground-truth time-shifts for dims 1-3
    delays = np.zeros(r_sim, dtype="int32")
    if r_sim > 1:
        delays[1] = np.random.randint(low=-max_shift, high=max_shift)
    if r_sim > 3:
        delays[3] = np.random.randint(low=-max_shift, high=max_shift)

    # Calibrating for 0-ms time-delay in 1 dimension
    delays[3] = 0

    # Time-shift latents
    Z0, Z1 = np.zeros_like(Z), np.zeros_like(Z)
    for dim, delay in enumerate(delays):
        f0 = _create_filter(delay, max_shift=max_shift)
        f1 = _create_filter(-delay, max_shift=max_shift)

        if dim == 3:
            f1 = _create_filter(1, max_shift=max_shift)

        Z0[:, dim] = np.convolve(Z[:, dim], f0, mode="same")
        Z1[:, dim] = np.convolve(Z[:, dim], f1, mode="same")

    # Make region-specific dimensions
    Z0_final = Z0.copy()
    Z1_final = Z1.copy()

    if r_sim > 0:
        Z0_final[:, -1] = 0
    if r_sim > 0:
        Z1_final[:, 0] = 0

    # Pad the latents to account for mSCA truncation
    padding = np.zeros((max_shift * 2 + 10, r_sim))
    Z0_padded = np.concatenate((padding, Z0_final, padding))
    Z1_padded = np.concatenate((padding, Z1_final, padding))

    # Project latents up into firing-rate space
    X0_fr, X1_fr = Z0_padded @ V0.T, Z1_padded @ V1.T

    # Can't have negative firing rates
    X0_fr -= X0_fr.min(axis=0)
    X1_fr -= X1_fr.min(axis=0)

    # Scale firing rates s.t. we get desired average firing rate across all "neurons"
    X0_fr *= avg_fr / X0_fr.mean()
    X1_fr *= avg_fr / X1_fr.mean()

    # Create trials by adding noise to high-D latents
    X0_trials, X1_trials = [], []
    for tri_num in range(n_trials):
        # Convert firing rates to spike counts drawn from a Poisson
        X0_trials.append(np.random.poisson(X0_fr * dts).astype("float32"))
        X1_trials.append(np.random.poisson(X1_fr * dts).astype("float32"))

    # Reformat for mSCA: {"Region #1 name": [trial_1, trial_2, ...]}
    X = {"X0": X0_trials, "X1": X1_trials}

    # Add noised neural activity to a dictionary
    X = {}
    X["X0"] = X0_trials
    X["X1"] = X1_trials

    # Add ground-truth latents to a dictionary
    Z_gt = {}
    Z_gt["Z0"] = Z0_final
    Z_gt["Z1"] = Z1_final

    delays_gt = delays.copy()

    return X, Z_gt, delays_gt
