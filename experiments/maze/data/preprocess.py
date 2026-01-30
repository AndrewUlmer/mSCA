import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt

import torch
from matplotlib import cm
from scipy import io
import time
from scipy.ndimage.filters import gaussian_filter1d

if __name__ == "__main__":
    # Load in the data
    load_folder = "/Users/andrewulmer/nuin/research/mSCA/experiments/maze/data/"
    data = io.loadmat(load_folder + "kaufman_trial_py2_good_fewertr")  # fewer neurons

    # I think I may have to make the filters = 21 instead of 31 here to avoid cutting off the target
    move_time_rel = data[
        "move_time_rel"
    ]  # time of movement relative to target (target always occurs at 30 bins here)
    neural_data = data["neural_data"]  # neural data
    condition = data[
        "condition"
    ]  # the trial condition number (there are 108 conditions)
    vels = data["vel"]  # velocities for each trial

    # Grab data for M1 and PMd separately
    array_idx = data["array_idx"]  ### Contains whether the unit was PMd (1) or M1 (2)
    m1_start = np.where(array_idx[0] == 2)[0][0]

    # Split into M1 / PMd
    m1_data = [x[m1_start:] for x in neural_data[0]]
    pmd_data = [x[:m1_start] for x in neural_data[0]]

    # Randomly sample 500 trials
    trial_idxs = np.random.randint(0, len(m1_data), size=500)
    m1_data = [m1_data[i].T for i in trial_idxs]
    pmd_data = [pmd_data[i].T for i in trial_idxs]

    # Save as a pytorch object
    X = {"M1": m1_data, "PMd": pmd_data}
    torch.save(X, f"{load_folder}x.pt")
