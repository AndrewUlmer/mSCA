import os
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
    # Set the experiment path
    experiment_path = "./experiments/cycling/mDLAG-comparison/"

    # Load the data
    fp = "./experiments/cycling/data/"
    m1_data, m1_mask, m1_start_idxs, m1_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_m1_rawRates.mat"
    )
    sma_data, sma_mask, sma_start_idxs, sma_end_idxs = load_data(
        f"{fp}", "Cousteau_interp_cycling_sma_rawRates.mat"
    )

    # Preprocess the data
    m1_preprocessed = preprocess(
        m1_data, m1_start_idxs, m1_end_idxs, downsample_factor=5
    )
    sma_preprocessed = preprocess(
        sma_data, sma_start_idxs, sma_end_idxs, downsample_factor=5
    )

    # Put data into dictionary format
    X = {"M1": m1_preprocessed, "SMA": sma_preprocessed}

    # Create splits across neurons and iteratively save
    if not os.path.exists(f"./{experiment_path}/n_train_splits.pt"):
        train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
        torch.save(train_idxs, f"./{experiment_path}/n_train_splits.pt")
        torch.save(test_idxs, f"./{experiment_path}/n_test_splits.pt")
    else:
        train_idxs = torch.load(f"./{experiment_path}/n_train_splits.pt")
        test_idxs = torch.load(f"./{experiment_path}/n_test_splits.pt")

    # Define the chunk size for training mDLAG
    chunk_size = 100

    for i, (train_m1, train_sma) in enumerate(zip(train_idxs["M1"], train_idxs["SMA"])):
        # empty container
        X_train = {}

        # grab the neural activity for the current training split
        X_train["M1"] = [x[:, train_m1] for x in X["M1"]]
        X_train["SMA"] = [x[:, train_sma] for x in X["SMA"]]

        # concatenate across regions for mDLAG
        x = [
            np.concatenate([x_m1, x_sma], axis=1)
            for x_m1, x_sma in zip(X_train["M1"], X_train["SMA"])
        ]

        # cut time-series into chunks for working with mDLAG
        x_chunked = []
        trial_id = []
        for trial_idx, trial in enumerate(x):
            chunks = np.array_split(
                trial, np.arange(chunk_size, trial.shape[0], chunk_size)
            )
            trial_id += [trial_idx + 1] * len(chunks)
            x_chunked += chunks

        # save the results as a matlab struct
        x_train_mat = {
            "y": x_chunked,
            "yDims": [X_train["M1"][0].shape[1], X_train["SMA"][0].shape[1]],
            "trialId": trial_id,
            "T": [len(x_i) for x_i in x_chunked],
        }
        io.savemat(
            f"{experiment_path}/x_train_split_{i}.mat",
            x_train_mat,
        )

        # torch.save(X_train, f"{experiment_path}/x_train_split_{i}.pt")
