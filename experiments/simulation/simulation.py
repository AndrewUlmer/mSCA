import torch
import matplotlib.pyplot as plt

from msca import *

# NOTE: we have manually made one of the time-delays in the ground-truth dataset
#       = 1 time bin, in order to showcase the high temporal resolution at which
#       mSCA performs.

# Generate noisy simulated firing-rates, ground-truth latents, and delays
X, Z_gt, delays_gt = simulate_single_trials()
# X, Z_gt, delays_gt = simulate_trial_averages_multi(num_pop=4)

# X = {"X": X["X0"]}

# Train the model
msca, losses = mSCA(
    n_components=5,
    n_epochs=10000,
    linear=False,
    loss_func="Poisson",
    cd_rate=0.5,
    filter_len=31,
).fit(X)

# Infer the latents
Z = msca.transform(X)

# Plot against the ground-truth
fig, axs = plt.subplots(5, 2, figsize=(5, 5))
for i in range(msca.n_components):
    axs[i, 0].plot(Z["X0"][0][:, i])
    axs[i, 0].plot(Z["X1"][0][:, i])
    axs[i, 0].plot(Z["X2"][0][:, i])
    axs[i, 0].plot(Z["X3"][0][:, i])

    axs[i, 1].plot(Z_gt["X0"][:, i])
    axs[i, 1].plot(Z_gt["X1"][:, i])
    axs[i, 1].plot(Z_gt["X2"][:, i])
    axs[i, 1].plot(Z_gt["X3"][:, i])

print("something")

# sparsity_range = [
#     0.0,
#     0.001,
#     0.01,
#     0.015,
#     0.02,
#     0.025,
#     0.03,
#     0.035,
#     0.04,
#     0.045,
#     0.05,
#     0.055,
#     0.06,
#     0.065,
#     0.07,
#     0.075,
#     0.08,
#     0.085,
#     0.09,
#     0.095,
#     0.1,
#     # 0.2,
#     # 0.3,
#     # 0.4,
#     # 0.5,
# ]
# pre = [
#     torch.load(
#         f"./experiments/simulation/sparsity_sweep_decoder_post_hoc/pre_{sparsity:.4f}.pt"
#     )
#     for sparsity in sparsity_range
# ]
# post = [
#     torch.load(
#         f"./experiments/simulation/sparsity_sweep_decoder_post_hoc/post_{sparsity:.4f}.pt"
#     )
#     for sparsity in sparsity_range
# ]

# print("something")
