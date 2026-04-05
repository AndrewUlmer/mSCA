import importlib
import os
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

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
    # Load the data
    print("Loading data...")
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

   
    print("making directories and file paths...")

    results_dir = "./experiments/cycling/results/Cousteau"
    os.makedirs(results_dir, exist_ok=True)

    model_path = os.path.join(results_dir, "msca_cycling_NonlinearEncoder_NonlinearDecoder_AdaptiveSparse.pt")
    loss_path = os.path.join(results_dir, "losses_NonlinearEncoder_NonlinearDecoder_AdaptiveSparse.npz")

    print(f"Model will be saved to: {model_path}")
    print(f"Losses will be saved to: {loss_path}")

    print("Fitting mSCA model...")
    msca,losses = mSCA(
        n_components=40,
        n_epochs=10000,
        linear=False, # nonlinear encoder 
        loss_func="Gaussian",
        lam_sparse="adaptive", # this is set to 0.05 in simulations -> this is not adaptive 
        #lam_orthog=0.5,
        lam_region=0.0,
        decoder_type="nonlinear", # nonlinear decoder
        decoder_hidden_size=40,
        decoder_activation="GeLU",
        post_hoc_epoch=-1,
        cd_rate=0.5,
        cd_mode="both",
        filter_len=41,
        init="unique",
        decoder_init_mode = "pca",
        #lam_sparse=0.1,
    ).fit(
        X
    )  # , load=True)


    
    msca.save(model_path)
    msca.save_losses(losses, loss_path)

    print(f"Saved model: {model_path}")
    print(f"Saved losses: {loss_path}")

  