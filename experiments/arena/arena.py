import os
import matplotlib.pyplot as plt
import numpy as np

import sys
import h5py
import os
import matplotlib.pyplot as plt

import sys

sys.path.append(os.getcwd())

from msca import *

def load_hdf5_to_dict(filename):
    """
    Recursive loader for the HDF5 files.
    """

    def recursive_load(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                data[key] = recursive_load(item)
            else:
                data[key] = item[()]
        return data

    with h5py.File(filename, "r") as h5file:
        return recursive_load(h5file)
    


    
data_dir = "./experiments/arena/data"
# Load preprocessed dataset
training_dataset = load_hdf5_to_dict(
    os.path.join(data_dir, "arena_training_bin_size=10.h5")
)

# Container for mSCA style input
X = {k: [] for k in ["cortex", "striatum"]}

# Split off only the striatal or cortical data
for brain_region in X.keys():
    for i, (k, v) in enumerate(training_dataset.items()):
        X[brain_region] += [v_i for v_i in v[brain_region]]

results_dir = "./experiments/arena/results"
os.makedirs(results_dir, exist_ok=True)
model_path = os.path.join(results_dir, "msca_nonlinearDecoder.pt") # 2000
loss_path = os.path.join(results_dir, "losses_nonlinearDecode.npz")


# Train mSCA: nolinear encoder, nonlinear decoder, default hyperparameters
msca, losses,_,_ = mSCA(
    n_components=20,
    loss_func="Poisson",
    n_epochs=5000,
    post_hoc_epoch=-1,
    linear=False,
    lam_region=0.0, # lam_sparse is adaptive so we need to turn on the warmup
    decoder_type="nonlinear", # nonlinear decoder
    decoder_hidden_size=40,
    decoder_activation="tanh",
    cd_rate=0.5,
    cd_mode="both",
    filter_len=41,
    init="unique",
    decoder_init_mode = "pca",
    sparsity_warmup_epochs=1000,
    balance_interval=1000,
 
).fit(X)


Z = msca.transform(X)
print("something")

# performances = bootstrap_delays_decoder(msca, X, num_bootstraps=100)
msca.save(model_path)
msca.save_losses(losses, loss_path)


print("something")