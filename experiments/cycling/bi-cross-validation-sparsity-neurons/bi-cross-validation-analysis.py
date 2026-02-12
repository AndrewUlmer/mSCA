import os
import sys
import shlex
import argparse
import linecache

import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from msca import *


def load_data(dir_path, f):
    # Load in matlab file
    mat_file = io.loadmat(dir_path + f)

    # Grab data and mask
    data_array = mat_file["data"]
    mask = mat_file["mask"]

    # Grab the start and end indices for each trial
    start_idxs = mat_file["mask"]["firstTimeEachCond"][0][0].squeeze() - 1
    end_idxs = mat_file["mask"]["lastTimeEachCond"][0][0].squeeze()

    return data_array, mask, start_idxs, end_idxs


# Preprocess the loaded data and split into trials
def preprocess(data_array, start_idxs, end_idxs, downsample_factor=5):
    """
    Pre-processing done by Andrew Zimnik
    """
    # Downsample the data
    if downsample_factor is not None:
        data_downsamp = data_array[::downsample_factor]

        # Adjust the start and end indices to account for the downsampling
        start_idxs = start_idxs // downsample_factor
        end_idxs = end_idxs // downsample_factor
    else:
        data_downsamp = data_array

    # Transpose the data to N x TC
    data_concat = data_downsamp.T

    # fr range (for normalizing later)
    fr_range = np.ptp(data_concat, axis=1)[:, None]

    # Soft normalize (divide each neuron by its range + 5)
    data_norm = data_concat / (fr_range + 5)

    # mean-center each neuron
    data_snm_norm = data_norm - np.mean(data_norm, axis=1)[:, None]

    # rename the data for convenience
    # Note that model requires (T x N) input rather than (N x T), which is why there are transposes below
    fit_data = np.copy(data_snm_norm.T)

    # Split the data into trials
    split_data = [fit_data[i:j] for i, j in zip(start_idxs, end_idxs)]

    return split_data


if __name__ == "__main__":
    # Set the experiment path
    experiment_path = "./experiments/cycling/bi-cross-validation-sparsity-neurons/"
    param_path = "params.txt"

    # Load the data
    fp = "./experiments/cycling/data/"
    m1_data, m1_mask, m1_start_idxs, m1_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_m1_rawRates.mat"
    )
    sma_data, sma_mask, sma_start_idxs, sma_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_sma_rawRates.mat"
    )

    # Set the experiment path
    experiment_path = "./experiments/cycling/bi-cross-validation-sparsity-neurons/"

    # Preprocess the data
    m1_preprocessed = preprocess(
        m1_data, m1_start_idxs, m1_end_idxs, downsample_factor=5
    )
    sma_preprocessed = preprocess(
        sma_data, sma_start_idxs, sma_end_idxs, downsample_factor=5
    )

    # Put data into dictionary format
    X = {"M1": m1_preprocessed, "SMA": sma_preprocessed}

    # Load folds for experiment
    train_idxs = torch.load(f"./{experiment_path}/n_train_splits.pt")
    test_idxs = torch.load(f"./{experiment_path}/n_test_splits.pt")

    # Iterate through the splits
    k0 = list(train_idxs.keys())[0]

    # Load all the sparsity settings from params.txt

    p = {}
    for sparsity in [
        0.0,
        0.001,
        0.01,
        0.1,
        1.0,
        1.5,
        2.0,
        2.5,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
    ]:  # [0.0, 0.001, 0.01, 0.1, 1.0, 2.5, 5.0]:  # 0.0, 0.001]:

        # Iterate over the different folds
        p_sparse = []
        for i in range(len(train_idxs[k0])):
            # Grab the neural data using the current training indices
            x_train = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}
            x_val = {k: [v_i[:, test_idxs[k][i]] for v_i in v] for k, v in X.items()}

            # Now run mSCA
            msca, _ = mSCA(
                n_components=40,
                n_epochs=10000,
                linear=True,
                loss_func="Gaussian",
                lam_sparse=10,  # 40=bad  # sparsity,
                lam_region=0.0,
                post_hoc_epoch=1000,
                cd_rate=0.5,
                cd_mode="both",
            ).fit(x_train, load=True)

            # Load the for this sparsity setting / neual split in
            msca.load(f"{experiment_path}msca_{i}_sparsity={sparsity:.4f}.pt", x_train)

            # Get splits for trials
            performances = evaluate_trial_average(msca, x_train, x_val, n_splits=5)

            # Iterate over splits
            p_sparse.append(performances)

        p[sparsity] = np.array(p_sparse)

    print("something")
