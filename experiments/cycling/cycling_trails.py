import os
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

    # Grab condition related things
    num_cycles = mask[0][0][1][start_idxs]
    direction = mask[0][0][2][start_idxs]
    position = mask[0][0][3][start_idxs]

    return data_array, mask, start_idxs, end_idxs, num_cycles, direction, position


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
    # Load the data
    fp = "./experiments/cycling/data/"
    m1_data, m1_mask, m1_start_idxs, m1_end_idxs, num_cycles, direction, position = (
        load_data(f"{fp}", "Drake_interp_cycling_m1_rawRates.mat")
    )
    sma_data, sma_mask, sma_start_idxs, sma_end_idxs, _, _, _ = load_data(
        f"{fp}", "Drake_interp_cycling_sma_rawRates.mat"
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

    # X = {"M1": [X["M1"][i] for i in range(4)], "SMA": [X["SMA"][i] for i in range(4)]}

    # Now run mSCA
    msca = mSCA(
        n_components=40,
        n_epochs=1000,  # 00,  # 0,
        loss_func="Gaussian",
        lam_region=0.5,
        post_hoc_epoch=-1,  # 1000,
        cd_rate=0.5,
        cd_mode="both",
    ).fit(X)

    # Infer latents
    Z = msca[0].transform(X)

    # Settings for plotting
    linestyles = {1: "-", -1: ":"}
    start_pos = {0.0: "r", 0.5: "b"}

    # Plot only M1
    fig, axs = plt.subplots(10, 4, figsize=(10, 7))

    ymax = np.concatenate(Z["SMA"], axis=0).max()
    ymin = np.concatenate(Z["SMA"], axis=0).min()

    for i in range(20):
        for j in range(40):
            axs[j // 4, j % 4].plot(
                Z["SMA"][i][:, j],
                c=start_pos[position[i].item()],
                ls=linestyles[direction[i].item()],
                alpha=0.5,
            )
            axs[j // 4, j % 4].ylim(ymin - 0.1, ymax + 0.1)

    print("something")