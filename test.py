import matplotlib.pyplot as plt
from msca import *

seed = 1

torch.manual_seed(seed)

# Generate noisy simulated firing-rates, ground-truth latents, and delays
X, Z_gt, delays_gt = simulate_trial_averages(random_seed=seed)

# Let's try now with simulated single-trials
# X, Z_gt, delays_gt = simulate_single_trials(random_seed=seed)

# Visualization + example accessing attributes of X
colors = ["#D81B60", "#1E88E5"]
num_viz = 5
fig, axs = plt.subplots(num_viz, 3, figsize=(10, 5))
for i in range(num_viz):
    axs[i, 0].plot(X["X0"][0][:, i], c=colors[0])
    axs[i, 1].plot(X["X1"][0][:, i], c=colors[1])
    axs[i, 2].plot(Z_gt["Z0"][:, i], c=colors[0])
    axs[i, 2].plot(Z_gt["Z1"][:, i], c=colors[1])

    if i == 0:
        axs[i, 0].set_title("Region 1 neural activity")
        axs[i, 1].set_title("Region 2 neural activity")
        axs[i, 2].set_title("Ground-truth latents")

    if i < num_viz - 1:
        axs[i, 0].set_xticks([])
        axs[i, 1].set_xticks([])
        axs[i, 2].set_xticks([])
    else:
        axs[i, 0].set_xlabel("time (ms)")
        axs[i, 1].set_xlabel("time (ms)")
        axs[i, 2].set_xlabel("time (ms)")

best_sparsity = sparsity_sweep_bootstrap(
    5 + 1, 3000, "Gaussian", X, "./check_delete_later/test_sweep/"
)

# msca, losses = mSCA(n_components=5 + 1, n_epochs=7000, loss_func="Poisson").fit(X)
# msca = mSCA(n_components=5 + 1, n_epochs=1)
# msca.fit(X)
# msca.load("./test.pt")

delay_effects = bootstrap_delays_decoder(msca, X)
msca_refined = refine_delays(msca, delay_effects)

# Grab the latents
Z = msca_refined.transform(X)

# Grab predictions
X_hat = msca.predict(X)

plt.clf()
loss = np.stack([v for v in losses.values()], axis=1).sum(axis=1)
colors = ["#D81B60", "#1E88E5"]
fig, axs = plt.subplots(5 + 1, 2, figsize=(10, 5))

for i in range(6):

    axs[i, 0].plot(Z["X0"][0][:, i], c=colors[0])
    axs[i, 0].plot(Z["X1"][0][:, i], c=colors[1])
    # axs[i, 0].set_ylim(-1.1, 1.1)

    if i < 5:
        axs[i, 1].plot(Z_gt["Z0"][:, i], c=colors[0])
        axs[i, 1].plot(Z_gt["Z1"][:, i], c=colors[1])
        # axs[i, 1].set_ylim(-1.1, 1.1)


# Grab the reconstructions
X_hat = msca.predict(X)

# Check the quality of the reconstructions
plt.clf()
plt.plot(X["X0"][0][msca.trunc, 0])
plt.plot(X_hat["X0"][0][:, 0])

print("something")
