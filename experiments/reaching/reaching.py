import matplotlib.pyplot as plt
from msca import *

# If you're trying to run this notebook without the data, this will fail
data = torch.load("./experiments/reaching/data/x.pt", weights_only=False)

# Grab only the neural data
X = {
    "M1": [x.astype("float32") for x in data["M1"]],
    "PMd": [x.astype("float32") for x in data["PMd"]],
}

# Remove weird neuron from pmd
X["PMd"] = [np.concatenate([x[:, :23], x[:, 23 + 1 :]], axis=1) for x in X["PMd"]]

# Train mSCA
msca, losses = mSCA(
    n_components=20, loss_func="Poisson", n_epochs=6000, post_hoc_epoch=5
).fit(X)

performances = bootstrap_delays_decoder(msca, X, num_bootstraps=100)


print("something")
