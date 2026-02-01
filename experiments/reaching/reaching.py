import os
import sys
import shlex
import argparse
import linecache

from msca import *
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

if __name__ == "__main__":
    ### IMPORTANT: ADD PATH TO YOUR params.txt
    experiment_path = "./experiments/reaching/sparsity_sweep_orth/"
    param_path = "params.txt"

    # IMPORTANT: ADD DATA LOADING CODE HERE
    data = torch.load("./experiments/reaching/data/x.pt", weights_only=False)
    X = {
        "M1": [x.astype("float32") for x in data["M1"]],
        "PMd": [x.astype("float32") for x in data["PMd"]],
    }

    performances = {k: [] for k in [0.01, 0.025, 0.05, 0.075, 0.1, 1.0]}
    for sparsity in [
        0.0,
        0.001,
        0.01,
        0.015,
        0.02,
        0.025,
        0.03,
        0.035,
        0.04,
        0.045,
        0.05,
        0.055,
        0.06,
        0.065,
        0.07,
        0.075,
        0.08,
        0.085,
        0.09,
        0.095,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]:

        # # Let's train mSCA with the current values
        # msca, losses = mSCA(
        #     n_components=20,
        #     lam_sparse=0.0,
        #     n_epochs=1,  # 8000,
        #     lam_region=0.0,
        #     loss_func="Poisson",
        # ).fit(X)

        # # # Load in the trained model
        # msca.load(f"{experiment_path}msca_n_components=20_lam_sparse={sparsity:.3f}.pt")

        # # Transform the latents for plotting purposes
        # Z = msca.transform(X)

        # # Plot the latents
        # fig, axs = plt.subplots(20, 8, figsize=(10, 10))

        # colors = [
        #     "#e60000",
        #     "#ff9900",
        #     "#663300",
        #     "#33cc00",
        #     "#00cccc",
        #     "#0000ff",
        #     "#a366ff",
        #     "#ff00ff",
        # ]

        # indices = [
        #     np.stack(
        #         [
        #             np.linalg.norm((z_m1 + z_pmd), axis=0)
        #             for z_m1, z_pmd in zip(Z["M1"], Z["PMd"])
        #         ]
        #     )[:, i]
        #     .argmax()
        #     .item()
        #     for i in range(20)
        # ]

        # for reach_dir in range(8):
        #     # Select a random trial from that reach direction
        #     # trial_idx = indices[reach_dir]
        #     trial_idx = np.random.choice(np.arange(500))
        #     min_bounds = np.minimum(
        #         np.concatenate(Z["M1"], axis=0).min(axis=0),
        #         np.concatenate(Z["PMd"], axis=0).min(axis=0),
        #     )
        #     max_bounds = np.maximum(
        #         np.concatenate(Z["M1"], axis=0).max(axis=0),
        #         np.concatenate(Z["PMd"], axis=0).max(axis=0),
        #     )
        #     for i, j in enumerate(range(20)):
        #         axs[i, reach_dir].plot(
        #             Z["M1"][trial_idx][:, j], color=colors[reach_dir]
        #         )
        #         axs[i, reach_dir].plot(
        #             Z["PMd"][trial_idx][:, j], color=colors[reach_dir], ls=":"
        #         )
        #         axs[i, reach_dir].set_ylim(min_bounds[j] - 0.1, max_bounds[j] + 0.1)

        #         if reach_dir > 0:
        #             axs[i, reach_dir].set_yticks([])
        #         if i < 11:
        #             axs[i, reach_dir].set_xticks([])

        # print("something")

        performance = torch.load(
            f"./experiments/reaching/sparsity_sweep_orth/bootstrapped_separate_n_components=20_lam_sparse={sparsity:.3f}.pt"
        )
        for alpha in [0.01, 0.025, 0.05, 0.075, 0.1, 1.0]:
            performances[alpha].append(performance[alpha])

    # Save performance
    patches = []
    for i, alpha in enumerate([0.01, 0.025, 0.05, 0.075, 0.1, 1.0]):
        plt.violinplot(
            [x for x in performances[alpha]],
            positions=[
                0.0,
                0.001,
                0.01,
                0.015,
                0.02,
                0.025,
                0.03,
                0.035,
                0.04,
                0.045,
                0.05,
                0.055,
                0.06,
                0.065,
                0.07,
                0.075,
                0.08,
                0.085,
                0.09,
                0.095,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            widths=0.005,
        )

        # The color matches the 'bodies' color specified above
        patches.append(mpatches.Patch(color=f"C{i}", label=f"alpha = {alpha}"))

        # Add the legend using the proxy artist
        plt.legend(handles=patches)
        plt.xlim(-0.01, 0.31)

    print("something")
