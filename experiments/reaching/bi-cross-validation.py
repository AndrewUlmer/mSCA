import os
import sys
import shlex
import linecache

import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from msca import *


def bi_cross_validation_neuron_indices(X, n_splits=5):
    """
    This will create bi-cross-validation indices (across neurons)

    X : dict
        Dictionary of neural data described in quickstart.ipynb
    n_splits : int
        The number of splits across neurons to do
    """

    # Get the number of neurons for each region
    k0 = list(X.keys())[0]
    if isinstance(X[k0], list):
        n_neurons = {k: v[0].shape[1] for k, v in X.items()}
    else:
        n_neurons = {k: v.shape[1] for k, v in X.items()}

    # Create random splits for each region
    idxs = {k: np.arange(v) for k, v in n_neurons.items()}
    bcv_train_idxs = {k: [] for k in X.keys()}
    bcv_test_idxs = {k: [] for k in X.keys()}
    for k in X.keys():
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
        for train_index, test_index in kf.split(idxs[k]):
            bcv_train_idxs[k].append(train_index)
            bcv_test_idxs[k].append(test_index)

    return bcv_train_idxs, bcv_test_idxs



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




# """
# 5-fold bi-cross-validation on reaching data — nonlinear encoder + linear decoder.
# """
# if __name__ == "__main__":
#     # Load reaching data
#     data = torch.load(
#         "./experiments/reaching/data/x_target_aligned.pt", weights_only=False
#     )

#     X = {
#         "M1":  [x.astype("float32") for x in data["M1"]],
#         "PMd": [x.astype("float32") for x in data["PMd"]],
#     }

#     # Output directory
#     experiment_path = "./experiments/reaching/results/bi-cross-validation/nonlinear-linear/"
#     os.makedirs(experiment_path, exist_ok=True)

#     # Create or load 5-fold neuron splits
#     train_split_path = os.path.join(experiment_path, "n_train_splits.pt")
#     test_split_path  = os.path.join(experiment_path, "n_test_splits.pt")
#     if not os.path.exists(train_split_path):
#         train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
#         torch.save(train_idxs, train_split_path)
#         torch.save(test_idxs,  test_split_path)
#     else:
#         train_idxs = torch.load(train_split_path, weights_only=False)
#         test_idxs  = torch.load(test_split_path,  weights_only=False)

#     k0 = list(train_idxs.keys())[0]

#     for i in range(len(train_idxs[k0])):
#         print(f"\n=== fold {i} ===")
#         # Subset to training neurons only
#         x = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}

#         # nonlinear encoder + linear decoder 
#         msca, losses,_,_ = mSCA(
#             n_components=20,
#             loss_func="Poisson",
#             n_epochs=5000,
#             post_hoc_epoch=-1,
#             linear=False,
#             lam_region=0.0, # lam_sparse is adaptive so we need to turn on the warmup
#             decoder_type="linear", # linear decoder
#             cd_rate=0.5,
#             cd_mode="both",
#             filter_len=41,
#             init="unique",
#             decoder_init_mode = "pca",
#             sparsity_warmup_epochs=-1,
#             balance_interval=100,
        
#         ).fit(x)

#         # # nonlinear encoder + nonlinear decoder
#         # msca, losses,_,_ = mSCA(
#         #     n_components=20,
#         #     loss_func="Poisson",
#         #     n_epochs=5000,
#         #     post_hoc_epoch=-1,
#         #     linear=False,
#         #     lam_region=0.0, # lam_sparse is adaptive so we need to turn on the warmup
#         #     decoder_type="nonlinear", # nonlinear decoder
#         #     decoder_hidden_size=40,
#         #     decoder_activation="tanh",
#         #     cd_rate=0.5,
#         #     cd_mode="both",
#         #     filter_len=41,
#         #     init="unique",
#         #     decoder_init_mode = "pca",
#         #     sparsity_warmup_epochs=1000,
#         #     balance_interval=1000,
        
#         # ).fit(x)



#         msca.save(os.path.join(experiment_path, f"msca_split_{i}.pt"))

#     print("\nDone")



# """

# This is sweeping over sparsity script to run bi-cross-validation on the Cousteau cycling data.
# """
# if __name__ == "__main__":
#     lam_sparsity_values = [
#     1e-3,
#     1e-2,
#     1e-1,
#     1,
#     1.5,
#     2,
#     2.5,
#     5,
#     10,
#     15,
#     20,
#     30,
#     40]
#     # Load the data
#     fp = "./experiments/cycling/data/"
#     m1_data, m1_mask, m1_start_idxs, m1_end_idxs = load_data(
#         f"{fp}", "Drake_interp_cycling_m1_rawRates.mat"
#     )
#     sma_data, sma_mask, sma_start_idxs, sma_end_idxs = load_data(
#         f"{fp}", "Drake_interp_cycling_sma_rawRates.mat"
#     )

#     # Set the experiment path
#     experiment_path = "./experiments/cycling/results/Drake/bi-cross-validation-sparsity/sweep/"
#     os.makedirs(experiment_path, exist_ok=True)

#     # Preprocess the data
#     m1_preprocessed = preprocess(
#         m1_data, m1_start_idxs, m1_end_idxs, downsample_factor=5
#     )
#     sma_preprocessed = preprocess(
#         sma_data, sma_start_idxs, sma_end_idxs, downsample_factor=5
#     )

#     # Put data into dictionary format
#     X = {"M1": m1_preprocessed, "SMA": sma_preprocessed}

#     # Load folds for experiment if they don't exist already
#     train_split_path = os.path.join(experiment_path, "n_train_splits.pt")
#     test_split_path  = os.path.join(experiment_path, "n_test_splits.pt")
#     if not os.path.exists(train_split_path):
#         train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
#         torch.save(train_idxs, train_split_path)
#         torch.save(test_idxs, test_split_path)
#     else:
#         train_idxs = torch.load(train_split_path)
#         test_idxs  = torch.load(test_split_path)


#     # Iterate through the splits
#     k0 = list(train_idxs.keys())[0]

#     for lam_sparsity in lam_sparsity_values:
#         model_dir = os.path.join(experiment_path, f"lam_sparsity_{lam_sparsity}")
#         os.makedirs(model_dir, exist_ok=True)

#         for i in range(len(train_idxs[k0])):
#             # Grab the neural data using the current training indices
#             x = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}

#             # Now run mSCA (nonlinear encoder + nonlinear decoder)
#             # adaptive sparsity with warmup
#             # no sparsity panelty 
#             # sweep for lam_sparse values
#             msca,losses,train_lambda_dicts, train_gradient_dicts = mSCA(
#             n_components=40,
#             n_epochs=5000,
#             linear=False, # nonlinear encoder 
#             loss_func="Gaussian",
#             lam_sparse=lam_sparsity, # this is set to 0.05 in simulations -> this is not adaptive 
#             lam_orthog=0.1,
#             lam_region=0.0,
#             decoder_type="nonlinear", # nonlinear decoder
#             decoder_hidden_size=40,
#             sparsity_warmup_epochs=-1,
#             decoder_activation="tanh",
#             post_hoc_epoch=-1,
#             cd_rate=0.5,
#             cd_mode="both",
#             filter_len=41,
#             init="unique",
#             decoder_init_mode = "pca",

#         ).fit(x)

#             # Save the mSCA model for the current split
#             msca.save(os.path.join(model_dir, f"msca_split_{i}.pt"))

#     print("Done")



"""
This is a script to run bi-cross-validation on the reaching data to generate a sweep over n_components values for three types of models
lam_sparse is fixed at 2.5, and lam_region is fixed at 0.0. 

"""

if __name__ == "__main__":
    # Fixed experiment settings
    #n_components_list = [6,10,16,20,30,40] # n_components to sweep over for reaching data 
    n_components_list = [6]
    # Base output path for nonlinear runs
    experiment_path = f"./experiments/reaching/results/sweep-n_components/new_nonlinear-linear"
    os.makedirs(experiment_path, exist_ok=True)

    # Load reaching data
    data = torch.load(
        "./experiments/reaching/data/x_target_aligned.pt", weights_only=False
    )

    X = {
        "M1":  [x.astype("float32") for x in data["M1"]],
        "PMd": [x.astype("float32") for x in data["PMd"]],
    }

    # Create or load 5-fold neuron splits
    train_split_path = os.path.join(experiment_path, "n_train_splits.pt")
    test_split_path  = os.path.join(experiment_path, "n_test_splits.pt")
    if not os.path.exists(train_split_path):
        train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
        torch.save(train_idxs, train_split_path)
        torch.save(test_idxs,  test_split_path)
    else:
        train_idxs = torch.load(train_split_path, weights_only=False)
        test_idxs  = torch.load(test_split_path,  weights_only=False)

    k0 = list(train_idxs.keys())[0]


    for n_components in n_components_list:
        model_dir = os.path.join(experiment_path, f"n_components_{n_components}")
        os.makedirs(model_dir, exist_ok=True)

        for i in range(len(train_idxs[k0])):
            # Grab the neural data using the current training indices
            x = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}

            # # nonlinear encoder + nonlinear decoder
            # msca, losses,_,_ = mSCA(
            #     n_components=n_components,
            #     loss_func="Poisson",
            #     n_epochs=5000,
            #     post_hoc_epoch=-1,
            #     linear=False,
            #     lam_region=0.0, # lam_sparse is adaptive so we need to turn on the warmup
            #     decoder_type="nonlinear", # nonlinear decoder
            #     decoder_hidden_size=40,
            #     decoder_activation="tanh",
            #     cd_rate=0.5,
            #     cd_mode="both",
            #     filter_len=41,
            #     init="unique",
            #     decoder_init_mode = "pca",
            #     sparsity_warmup_epochs=1000,
            #     balance_interval=1000,
            
            # ).fit(x)



            # nonlinear encoder + linear decoder 
            msca, losses,_,_ = mSCA(
                n_components=n_components,
                loss_func="Poisson",
                n_epochs=5000,
                post_hoc_epoch=-1,
                linear=False,
                lam_region=0.0, # lam_sparse is adaptive so we need to turn on the warmup
                decoder_type="linear", # linear decoder
                cd_rate=0.5,
                cd_mode="both",
                filter_len=41,
                init="unique",
                decoder_init_mode = "pca",
                sparsity_warmup_epochs=-1,
                balance_interval=100,
            
            ).fit(x)

            # Save model and training losses for this split
            msca.save(os.path.join(model_dir, f"msca_split_{i}.pt"))
            torch.save(losses, os.path.join(model_dir, f"losses_split_{i}.pt"))

    print("Done.")





# """
# This is a script to run bi-cross-validation on the Cousteau cycling data to generate a sweep over lam_region values for a nonlinear encoder + nonlinear decoder model
# lam_sparse is fixed at 2.5, and n_components is fixed at 40 or 70."""
# 70
# if __name__ == "__main__":
#     # Fixed experiment settings
#     n_components = 70
#     lam_sparse = 2.5
#     lam_region_sparse = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.6, 2.8, 3.0]
 
#     # Base output path
#     experiment_root = (
#         "./experiments/cycling/results/"
#         "bi-cross-validation-Cousteau-nonlinear-nonlinear-lamsparse-2.5-components-70/"
#     )
#     os.makedirs(experiment_root, exist_ok=True)

#     # Load the data
#     fp = "./experiments/cycling/data/"
#     m1_data, m1_mask, m1_start_idxs, m1_end_idxs = load_data(
#         f"{fp}", "Cousteau_interp_cycling_m1_rawRates.mat"
#     )
#     sma_data, sma_mask, sma_start_idxs, sma_end_idxs = load_data(
#         f"{fp}", "Cousteau_interp_cycling_sma_rawRates.mat"
#     )

#     # Preprocess the data
#     m1_preprocessed = preprocess(
#         m1_data, m1_start_idxs, m1_end_idxs, downsample_factor=5
#     )
#     sma_preprocessed = preprocess(
#         sma_data, sma_start_idxs, sma_end_idxs, downsample_factor=5
#     )

#     # Put data into dictionary format
#     X = {"M1": m1_preprocessed, "SMA": sma_preprocessed}

#     # Shared fold files
#     train_split_path = os.path.join(experiment_root, "n_train_splits.pt")
#     test_split_path = os.path.join(experiment_root, "n_test_splits.pt")

#     # Load folds for experiment if they don't exist already
#     if not os.path.exists(train_split_path):
#         train_idxs, test_idxs = bi_cross_validation_neuron_indices(X, n_splits=5)
#         torch.save(train_idxs, train_split_path)
#         torch.save(test_idxs, test_split_path)
#     else:
#         train_idxs = torch.load(train_split_path)
#         test_idxs = torch.load(test_split_path)

#     # Iterate through region sparsity settings
#     k0 = list(train_idxs.keys())[0]

#     for lam_region in lam_region_sparse:
#         lam_region_str = str(lam_region).replace(".", "p")
#         model_dir = os.path.join(
#             experiment_root,
#             f"lam_region_{lam_region_str}",
#         )
#         os.makedirs(model_dir, exist_ok=True)

#         # Save config for this sweep value
#         torch.save(
#             {
#                 "n_components": n_components,
#                 "lam_sparse": lam_sparse,
#                 "lam_region": lam_region,
#                 "decoder_type": "nonlinear",
#                 "linear": False,
#                 "loss_func": "Gaussian",
#             },
#             os.path.join(model_dir, "config.pt"),
#         )

#         print(f"\n=== lam_region={lam_region} ===")

#         for i in range(len(train_idxs[k0])):
#             # Grab the neural data using the current training indices
#             x = {k: [v_i[:, train_idxs[k][i]] for v_i in v] for k, v in X.items()}

#             # Nonlinear encoder + nonlinear decoder
#             msca, losses = mSCA(
#                 n_components=n_components,
#                 n_epochs=5000,
#                 linear=False,
#                 loss_func="Gaussian",
#                 lam_sparse=lam_sparse,
#                 lam_orthog=0.0,
#                 lam_region=lam_region,
#                 decoder_type="nonlinear",
#                 decoder_hidden_size=n_components,
#                 decoder_activation="GeLU",
#                 post_hoc_epoch=-1,
#                 cd_rate=0.5,
#                 cd_mode="both",
#                 filter_len=41,
#                 init="unique",
#                 decoder_init_mode="pca",
#             ).fit(x)

#             # Save model and training losses for this split
#             msca.save(os.path.join(model_dir, f"msca_split_{i}.pt"))
#             torch.save(losses, os.path.join(model_dir, f"losses_split_{i}.pt"))

#             print(f"  saved split {i} for lam_region={lam_region}")

#     print("Done.")
