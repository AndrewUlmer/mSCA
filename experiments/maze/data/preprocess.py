import numpy as np
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import io
import torch
from scipy.ndimage.filters import gaussian_filter1d


if __name__ == "__main__":
    load_folder = "/Users/andrewulmer/nuin/research/mSCA/experiments/maze/data/"
    data = io.loadmat(load_folder + "kaufman_trial_py2_good_fewertr")  # fewer neurons

    move_time_rel = data[
        "move_time_rel"
    ]  # time of movement relative to target (target always occurs at 30 bins here)
    neural_data = data["neural_data"]  # neural data
    condition = data[
        "condition"
    ]  # the trial condition number (there are 108 conditions)
    vels = data["vel"]  # velocities for each trial

    # Grab index for separating regions
    array_idx = data["array_idx"]  ### Contains whether the unit was PMd (1) or M1 (2)
    m1_start = np.where(array_idx[0] == 2)[0][0]

    # Separate m1 and pmd
    m1_data = [x[m1_start:] for x in neural_data[0]]
    pmd_data = [x[:m1_start] for x in neural_data[0]]

    # Randomly sample trials
    num_trials = len(m1_data)
    trial_idxs = np.random.randint(0, num_trials, 500)
    m1_data = [m1_data[i].T for i in trial_idxs]
    pmd_data = [pmd_data[i].T for i in trial_idxs]

    # Grab the movement times
    move_times = [int(move_time_rel[0][i].item() / 10) + 30 for i in trial_idxs]

    # Grab the target times
    tgt_times = [30] * 500

    # Save x
    X = {"M1": m1_data, "PMd": pmd_data, "move_idxs": move_times, "tgt_idxs": tgt_times}

    torch.save(X, f"{load_folder}x.pt")
    print("something")
