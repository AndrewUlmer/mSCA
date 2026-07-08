import os
import matplotlib.pyplot as plt

import sys

sys.path.append(os.getcwd())

from msca import *

# If you're trying to run this notebook without the data, this will fail
data = torch.load(
    "./experiments/reaching/data/x_target_aligned.pt", weights_only=False
)  # _long_2.pt", weights_only=False)

# Grab only the neural data
X = {
    "M1": [x.astype("float32") for x in data["M1"]],
    "PMd": [x.astype("float32") for x in data["PMd"]],
}

results_dir = "./experiments/reaching/results"
os.makedirs(results_dir, exist_ok=True)
model_path = os.path.join(results_dir, "msca_nonlinearDecoder_warmup_1000_balance_1000.pt") # 2000
loss_path = os.path.join(results_dir, "losses_nonlinearDecoder_warmup_1000_balance_1000.npz")


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