import os
import sys
import shlex
import argparse
import linecache

import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from msca import *


def get_params_by_id(path_to_file):
    # Get the Array ID from Slurm -> default to 1
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    # Read the specific line from the file
    line = linecache.getline(path_to_file, task_id).strip()

    if not line:
        print(f"Error: Line {task_id} in {path_to_file} is empty/missing.")
        sys.exit(1)

    print(f"Worker #{task_id} processing args: {line}")

    # Split the string into a list (handles quotes correctly)
    return shlex.split(line)


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
    # Grab parameters passed via cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam_sparse", type=float, required=True)
    parser.add_argument("--n_components", type=int, required=True)

    # Set the experiment path
    experiment_path = "./experiments/cycling/bi-cross-validation/"
    param_path = "params.txt"

    # Read the specific parameters from the text file
    file_args = get_params_by_id(experiment_path + param_path)

    # Load them in and convert to dictionary
    args = parser.parse_args(file_args)
    args = vars(args)

    # Load the data
    fp = "./experiments/cycling/data/"
    m1_data, m1_mask, m1_start_idxs, m1_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_m1_rawRates.mat"
    )
    sma_data, sma_mask, sma_start_idxs, sma_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_sma_rawRates.mat"
    )

    # Set the experiment path
    experiment_path = "./experiments/cycling/bi-cross-validation/"

    # Preprocess the data
    m1_preprocessed = preprocess(
        m1_data, m1_start_idxs, m1_end_idxs, downsample_factor=5
    )
    sma_preprocessed = preprocess(
        sma_data, sma_start_idxs, sma_end_idxs, downsample_factor=5
    )

    # Put data into dictionary format
    X = {"M1": m1_preprocessed, "SMA": sma_preprocessed}

    # Load folds for experiment if they don't exist already
    if not os.path.exists(f"./{experiment_path}/n_train_splits.pt"):
        train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
        torch.save(train_idxs, f"./{experiment_path}/n_train_splits.pt")
        torch.save(test_idxs, f"./{experiment_path}/n_test_splits.pt")
    else:
        train_idxs = torch.load(f"./{experiment_path}/n_train_splits.pt")
        test_idxs = torch.load(f"./{experiment_path}/n_test_splits.pt")

    # Iterate through the splits
    k0 = list(train_idxs.keys())[0]

    for i in range(len(train_idxs[k0])):
        # Grab the neural data using the current training indices
        x = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}

        # Now run mSCA
        msca, losses = mSCA(
            n_components=40,
            n_epochs=10000,
            linear=True,
            loss_func="Gaussian",
            lam_sparse=args["lam_sparse"],
            lam_region=0.0,
            post_hoc_epoch=1000,
            cd_rate=0.5,
            cd_mode="both",
        ).fit(x)

        # Save the mSCA model for the current split
        msca.save(f"{experiment_path}msca_{i}_sparsity={args['lam_sparse']:.4f}.pt")
        
        # Save the losses
        torch.save(losses, f"{experiment_path}losses_{i}_sparsity={args['lam_sparse']:.4f}.pt")

    print("something")
