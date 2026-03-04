import matplotlib.pyplot as plt
from msca import *

# If you're trying to run this notebook without the data, this will fail
data = torch.load("./experiments/reaching/data/x_long_2.pt", weights_only=False)

# Note: neuron 14 in M1 is one that loads heavily onto super early M1 --> PMd signal

# Grab only the neural data
X = {
    "M1": [x.astype("float32") for x in data["M1"]],
    "PMd": [x.astype("float32") for x in data["PMd"]],
}

# Let's look at weird neuron in M1 - make empty guy
# empties_n, empties_k = [], []
# for i in range(84):
#     empty_n = np.zeros(max([x_i.shape[0] for x_i in X["M1"]]))
#     empty_k = np.zeros(max([x_i.shape[0] for x_i in X["M1"]]))
#     for j, x_i in enumerate(X["PMd"]):
#         empty_n[: len(x_i)] += x_i[:, i]
#         empty_k[: len(x_i)] += np.sqrt(
#             data["kinematics"][j][:, 2] ** 2 + data["kinematics"][j][:, 3] ** 2
#         )

#     empties_n.append(empty_n)
#     empties_k.append(empty_k)

# [(plt.subplot(4, 21, i + 1), plt.plot(empties_n[i])) for i in range(84)]
# plt.tight_layout()
# print("something")

# Remove weird neuron from pmd
# X["PMd"] = [np.concatenate([x[:, :23], x[:, 23 + 1 :]], axis=1) for x in X["PMd"]]
# X["M1"] = [np.concatenate([x[:, :14], x[:, 14 + 1 :]], axis=1) for x in X["M1"]]

# Train mSCA
msca, losses = mSCA(
    n_components=20,
    loss_func="Poisson",
    n_epochs=10000,
    post_hoc_epoch=1000,
    filter_len=31,
    linear=False,
    lam_orthog=0.0,
    region_weights={"M1": 1.0, "PMd": 1.0},
).fit(X)

Z = msca.transform(X)
print("something")

performances = bootstrap_delays_decoder(msca, X, num_bootstraps=100)


print("something")
