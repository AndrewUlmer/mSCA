import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import PoissonRegressor
from sklearn.multioutput import MultiOutputRegressor

from .loss_funcs import *

from .models import *


def sparsity_sweep_bootstrap(
    n_components: int, n_epochs: int, loss_func: str, X: dict, path: str
):
    """
    This performs a sweep over the sparsity hyperparameter and
    saves the results to a folder specified in path

    Parameters
    ----------
    n_components : int
        Number of latent factors to use for each fit
    n_epochs : int
        Number of epochs to fit each model with
    loss_func : str
        Either "Gaussian" or "Poisson - the loss function used
        to train mSCA
    X : dict
        Dictionary of neural acitivity described in quickstary.ipynb
    path : str
        This is the path you want to save your trained models to.
    """
    print(f"Performing sparsity sweep 🧹 --> saving results to {path}")
    performances = {}

    # Use different ranges for different loss functions
    if loss_func == "Gaussian":
        sparsity_range = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    elif loss_func == "Poisson":
        sparsity_range = np.array([0.0, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1.0])

    for sparsity in sparsity_range:
        # Correcting weird np to python conversion
        sparsity = float(f"{sparsity:0.3f}")

        # Instantiate mSCA with desired sparsity level
        msca = mSCA(
            n_components=n_components,
            n_epochs=n_epochs,
            loss_func=loss_func,
            lam_sparse=sparsity,
        )
        msca, losses = msca.fit(X)

        # Perform bootstrap validation
        bootstrapped_losses = bootstrap_performances(msca, X)

        # Store the performances
        performances[sparsity] = bootstrapped_losses

        # Save the bootstrapped model performance
        torch.save(
            bootstrapped_losses,
            f"{path}/bootstrapped_sparsity={sparsity:.4f}.pt",
        )

        # Save the model so we can retrieve the best model later
        msca.save(f"{path}/msca_sparsity={sparsity:.4f}.pt")

        # Save the losses (for checking convergence)
        torch.save(losses, f"{path}/losses_sparsity={sparsity:.4f}.pt")

    return performances


@torch.no_grad()
def bootstrap_performances(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 1000
) -> dict[int, np.ndarray]:
    """
    This reconstructs the neural activity randomly ablating neurons and
    time-points to induce a distribution over loss values.
    It uses mSCA's learned decoder matrix

    Parameters
    ----------
    msca : mSCA object
        A trained instantiation of mSCA
    X : dict[str, np.ndarray]
        Format described in quickstart.ipynb
    num_bootstraps : int
        Number of bootstraps to perform
    """
    # Set the criterion for evaluation
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Convert X into a data_loader
    data_loader, _ = convert_to_dataloader(X)

    # Repeat for num_bootstraps
    bootstrapped_r2s = []
    for _ in tqdm(range(num_bootstraps)):

        # Used for storing reconstructions
        full_reconstruction = {k: [] for k in X.keys()}

        # Used for storing truncated inputs
        full_x_target = {k: [] for k in X.keys()}

        # Now iterate through trials in the data_loader
        for _, (X_target, trial_length) in enumerate(data_loader):
            # Apply the mask to the inputs and outputs
            X_input_masked, _, _, _, _ = msca.cd.forward(
                X_target,
                trial_length,
            )
            # Perform a forward pass through the model
            _, _, X_reconstruction = msca.model(X_input_masked)

            # Remove the padding from the reconstruction
            for k, v in X_reconstruction.items():
                for i, t in enumerate(trial_length):
                    X_reconstruction[k][i, (t - msca.filter_len) :] = 0

            # Add to full reconstructuion
            [full_reconstruction[k].append(X_reconstruction[k]) for k in X.keys()]
            [full_x_target[k].append(X_target[k][:, msca.trunc]) for k in X.keys()]

        # Concatenate reconstruction and masked target across batches
        full_reconstruction = {
            k: torch.cat(v, axis=0) for k, v in full_reconstruction.items()
        }
        full_x_target = {k: torch.cat(v, axis=0) for k, v in full_x_target.items()}

        # Concatenate recnonstruction and target across regions
        full_reconstruction = torch.cat(
            [v for v in full_reconstruction.values()], axis=2
        )  # type: ignore
        full_x_target = torch.cat([v for v in full_x_target.values()], axis=2)  # type: ignore

        # Compute the resulting r2 score, using normal r2 for Gaussian, pseudo-r2 for Poisson
        if msca.loss_func == "Gaussian":
            r2s = r2_score(full_x_target.flatten(), full_reconstruction.flatten())
        elif msca.loss_func == "Poisson":
            # Compute the "null model"
            mean_fr = full_x_target.mean(axis=(0, 1))

            # Collapse the batch (trial) dimension
            full_x_target = full_x_target.reshape(-1, full_x_target.shape[-1])
            full_reconstruction = full_reconstruction.reshape(
                -1, full_reconstruction.shape[-1]
            )

            # Compute pseudo r2
            r2s = pseudo_r2(full_x_target, full_reconstruction, mean_fr)

        bootstrapped_r2s.append(r2s)

    return np.array(bootstrapped_r2s)


def pseudo_r2(
    X_target: torch.Tensor, predictions: torch.Tensor, mean_fr: torch.Tensor
) -> float:
    """
    This function will compute the pseudo r2 using the NLL from the Poisson
    distribution

    Parameters
    ----------
    X_target : torch.Tensor
        The neural activity being reconstructed, concatenated across time and regions
    predictions : torch.Tensor
        The predictions made by mSCA's decoder, again concatenated across time and regions
    mean_fr : torch.Tensor
        The mean firing-rate of each neuron computed across all trials. Used to compute
        the negative log-likelihood for the null model

    """
    # Compute log-likelihood for saturated model
    sat_ll = poisson_f(X_target, X_target)

    # Compute the null log-likelihood
    null_ll = poisson_f(mean_fr, X_target)

    # Compute the actual log-likelihood
    ll = poisson_f(predictions, X_target)

    # Compute the pseudo-r2 - note these are NLLs
    D_model = ll.sum() - sat_ll.sum()
    D_null = null_ll.sum() - sat_ll.sum()
    r2 = 1 - (D_model / D_null)

    return r2


@torch.no_grad()
def evaluate_trial_average(msca, X_train, X_val, n_splits=5):
    """
    This is used to evaluate a bi-cross-validation fold, when actually
    holding out neurons

    Parameters
    ----------
    msca : object
        A trained instantiation of mSCA
    X_train : dict
        A dictionary containing neural activity used in the training set. Should
        be the same format as described in quickstart.ipynb
    X_val : dict
        A dictionary containing the neural activity used in the validation set.
    n_splits : int
        Number of splits to use across time.
    """
    # Compute the latents on all the training neurons
    Z = msca.transform(X_train)

    # Store variable for timepoints
    T = np.arange(len(Z[list(Z.keys())[0]]))

    # Initialize the KFold object across trials
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    criterion = eval(f"{msca.loss_func}_f".lower())

    # Iterate over the folds
    performances_all = []  # {k: [] for k in X_train.keys()}
    for train_index, test_index in kf.split(T):

        X_val_concat = {
            k: np.concatenate([x[msca.trunc] for x in v], axis=0)
            for k, v in X_val.items()
        }

        # Grab the latents on training trials
        Z_concat_train = {
            k: np.concatenate([v[i] for i in train_index], axis=0) for k, v in Z.items()
        }

        # Grab the latents on testing trials
        Z_concat_test = {
            k: np.concatenate([v[i] for i in test_index], axis=0) for k, v in Z.items()
        }

        # Grab the validation neurons on training trials
        X_val_concat_train = {
            k: np.concatenate([v[i][msca.trunc] for i in train_index], axis=0)
            for k, v in X_val.items()
        }

        # Grab the validation neuron on testing trials
        X_val_concat_test = {
            k: np.concatenate([v[i][msca.trunc] for i in test_index], axis=0)
            for k, v in X_val.items()
        }

        predictions, X_test_heldout = [], []
        for k in Z_concat_train.keys():
            # Grab the latents on the training/testing time points
            Z_train = Z_concat_train[k]
            Z_test = Z_concat_test[k]

            # # Grab the held-out neurons on the training/testing time points
            X_train_heldout = X_val_concat_train[k]
            X_test_heldout_r = X_val_concat_test[k]

            # Fit a linear regression model to map from training latents to held-out neurons
            lr = LinearRegression()
            lr.fit(Z_train, X_train_heldout)

            # Predict held-out neurons on held-out time-points
            predictions_r = lr.predict(Z_test)

            if msca.loss_func == "Poisson":
                predictions = np.maximum(predictions_r, 0)

            predictions.append(predictions_r)
            X_test_heldout.append(X_test_heldout_r)

        predictions = np.concatenate(predictions, axis=1)
        X_test_heldout = np.concatenate(X_test_heldout, axis=1)
        loss = r2_score(X_test_heldout, predictions)
        performances_all.append(loss)

    return performances_all


def mean_confidence_interval(data: np.ndarray, confidence: float = 0.95):
    """
    Simply computes a one-sides confidence interval based on the bootstrap results
    from bootstrap_delays_decoder

    Parameters
    ----------
    data : np.ndarray
        The bootstraps
    confidence : float
        the percentile to use for computing the confidence interval
    """
    m = np.mean(data)
    lower = np.percentile(data, 100 * (1 - confidence))
    return m, lower


@torch.no_grad()
def refine_delays(
    msca: object, bootstrapped_delay_effects: dict[int, np.ndarray], confidence=0.95
):
    """
    This removes those time-delays which do not meaningfully improve the reconstruction
    performance of the model.

    Parameters
    ----------
    msca : object
        Trained mSCA object
    bootstrapped_delay_effects : dict[int, np.ndarray]
        Bootstrapped delay effects output by bootstrap_delays_decoder
    confidence : float
        Will delete a delay if it doesn't improve reconstruction performance on 95%
        of the bootstraps.
    """
    # Iterate through the confidence intervals
    for i in range(msca.n_components):
        # Check if the delay effect is roughly significant
        significant = (
            mean_confidence_interval(
                bootstrapped_delay_effects[i], confidence=confidence
            )[1]
            > 0
        )

        # Check if the lower bound is less than 0
        if not significant:
            msca.model.filters.mus.data[i] = 0
    return msca


@torch.no_grad()
def bootstrap_delays_decoder(
    msca: object,
    X: dict[str, np.ndarray],
    num_bootstraps: int = 1000,
    mode: str = "both",
) -> dict[int, np.ndarray]:
    """
    This reconstructs the neural activity with and without each dimensions'
    time-delay and bootstraps over the differences in the loss function
    after deleting the time-delay. If the loss increases after deleting
    the delay, then that delay is important for reconstructing the data.

    Parameters
    ----------
    msca : mSCA object
        A trained instantiation of mSCA
    X : dict[str, np.ndarray]
        Format described in quickstart.ipynb
    num_bootstraps : int
        Number of bootstraps to perform
    """
    # Set the cd_rate
    msca.cd.cd_rate = 0.5

    # Set the criterion for evaluation
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Convert X into a data_loader
    data_loader, _ = convert_to_dataloader(X, shuffle=False)

    # Check loss func and convert cd if necessary
    if mode == "neurons":
        msca.cd.mode = "neurons"
    elif mode == "both":
        msca.cd.mode = "both"

    # Iterate through delays for each dimension
    performances = {}
    for i in tqdm(range(msca.n_components)):

        # Repeat for num_bootstraps
        diffs = []
        for _ in tqdm(range(num_bootstraps)):
            # Now iterate through trials in the data_loader
            with_delay, without_delay = 0, 0
            for _, (X_target, trial_length) in enumerate(data_loader):
                # Apply the mask to the inputs and outputs
                X_input_masked, X_output_masked, output_mask, _, _ = msca.cd.forward(
                    X_target,
                    trial_length,
                )

                # Forward pass with time-delay
                _, _, X_reconstruction_with_delay = msca.model(X_input_masked)

                # Forward pass without time-delay
                delay = msca.model.filters.mus[i].data.clone()
                msca.model.filters.mus[i] = 0
                _, _, X_reconstruction_without_delay = msca.model(X_input_masked)

                # Apply the output mask to the reconstructions
                X_reconstruction_with_delay_masked = msca.cd.mask(
                    X_reconstruction_with_delay, truncate(output_mask, msca.trunc)
                )
                X_reconstruction_without_delay_masked = msca.cd.mask(
                    X_reconstruction_without_delay, truncate(output_mask, msca.trunc)
                )

                # Compute the reconstruction loss with the time-delay
                with_delay += sum(
                    reconstruction_loss(
                        X_reconstruction_with_delay_masked,
                        truncate(X_output_masked, msca.trunc),
                        criterion,
                        mode="train",
                    )
                )

                # Compute the reconstruction loss without the time-delay
                without_delay += sum(
                    reconstruction_loss(
                        X_reconstruction_without_delay_masked,
                        truncate(X_output_masked, msca.trunc),
                        criterion,
                        mode="train",
                    )
                )

                # Reset delay
                msca.model.filters.mus.data[i] = delay

            # Compute the percent difference in the loss with/without the delay
            diffs.append(100 * (without_delay - with_delay) / abs(with_delay))

        performances[i] = np.array(diffs)

    ### TESTING: ADDING CODE FOR DELETING DELAYS FOR REGION-SPECIFIC DIMENSIONS

    # Compute the latents
    Z = msca.transform(X)

    # Compute their magnitudes
    magnitudes = np.stack(
        [np.linalg.norm(np.concatenate(v), axis=0) for v in Z.values()]
    )

    # If one dimension's magnitude is 90% of the totaly magnitude, zero-out delays for other dims
    region_specific = ((magnitudes / magnitudes.sum(axis=0)) > 0.9).sum(axis=0)
    for k, v in performances.items():
        if region_specific[k]:
            performances[k] = np.zeros_like(v)

    ##### END TESTING

    return performances


##### START CODE STILL IN DEVELOPMENT / OLD CODE #####


class PoissonRegressorWrapper:
    def __init__(self, alpha):
        self.alpha = alpha
        return

    def fit(self, Z, X):
        regressor = PoissonRegressor(alpha=self.alpha, solver="newton-cholesky")
        self.model = MultiOutputRegressor(regressor)
        self.model.fit(Z, X)
        return self

    def predict(self, Z):
        return self.model.predict(Z)


@torch.no_grad()
def bootstrap_latents_decoder(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 1000
) -> dict[int, np.ndarray]:
    """
    This reconstructs the neural activity with and without each dimension
    and bootstraps over the differences in the loss function after
    deleting the latent. If the loss increases after deleting
    the latent, then that latent is important for reconstructing the data.

    Parameters
    ----------
    msca : mSCA object
        A trained instantiation of mSCA
    X : dict[str, np.ndarray]
        Format described in quickstart.ipynb
    num_bootstraps : int
        Number of bootstraps to perform
    """
    # Set the criterion for evaluation
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Convert X into a data_loader
    data_loader, _ = convert_to_dataloader(X, shuffle=False)

    # Iterate through delays for each dimension
    performances = {}
    for i in tqdm(range(msca.n_components)):

        # Repeat for num_bootstraps
        diffs = []
        for _ in tqdm(range(num_bootstraps)):
            # Now iterate through trials in the data_loader
            with_latent, without_latent = 0, 0
            for _, (X_target, trial_length) in enumerate(data_loader):
                # Apply the mask to the inputs and outputs
                X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                    msca.cd.forward(
                        X_target,
                        trial_length,
                    )
                )

                # Forward pass with time-delay
                _, _, X_reconstruction_with_latent = msca.model(X_input_masked)

                # Forward pass without latent
                c = msca.model.decoder_scaling[i].data.clone()
                msca.model.decoder_scaling[i] = 0
                _, _, X_reconstruction_without_latent = msca.model(X_input_masked)

                # Apply the output mask to the reconstructions
                X_reconstruction_with_latent_masked = msca.cd.mask(
                    X_reconstruction_with_latent, truncate(output_mask, msca.trunc)
                )
                X_reconstruction_without_latent_masked = msca.cd.mask(
                    X_reconstruction_without_latent, truncate(output_mask, msca.trunc)
                )

                # Compute the reconstruction loss with the time-delay
                with_latent += sum(
                    reconstruction_loss(
                        X_reconstruction_with_latent_masked,
                        truncate(X_output_masked, msca.trunc),
                        criterion,
                        mode="train",
                    )
                )

                # Compute the reconstruction loss without the time-delay
                without_latent += sum(
                    reconstruction_loss(
                        X_reconstruction_without_latent_masked,
                        truncate(X_output_masked, msca.trunc),
                        criterion,
                        mode="train",
                    )
                )

                # Reset delay
                msca.model.decoder_scaling[i] = c

            # Compute the percent difference in the loss with/without the delay
            diffs.append(100 * (without_latent - with_latent) / with_latent.abs())

        performances[i] = np.array(diffs)

    return performances


@torch.no_grad()
def bootstrap_performances_separate_regressor(
    msca: object,
    X: dict[str, np.ndarray],
    alpha: float = 0.0,
    num_bootstraps: int = 1000,
    threshold: float = 0.1,  # 0.01
    mode: str = "both",
) -> dict[int, np.ndarray]:
    """
    This reconstructs the neural activity randomly ablating neurons and
    time-points to induce a distribution over loss values. It uses linear
    regression instead of the learned decoder matrix.

    Parameters
    ----------
    msca : mSCA object
        A trained instantiation of mSCA
    X : dict[str, np.ndarray]
        Format described in quickstart.ipynb
    num_bootstraps : int
        Number of bootstraps to perform
    mode : str
        Whether to bootstrap over both neurons and time-points,
        or just neurons
    """
    # Set the criterion for evaluation
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Infer latents for all the trials
    Z = msca.transform(X)

    # Concatenate latents and corresponding neural activity across all trials
    Z_full = {k: np.concatenate(z) for k, z in Z.items()}
    X_target_full = {
        k: np.concatenate([x_i[msca.trunc] for x_i in x]) for k, x in X.items()
    }

    # Fit decoders for both regions
    if msca.loss_func == "Poisson":
        regressor = {
            k: PoissonRegressorWrapper(alpha).fit(Z_full[k], X_target_full[k])
            for k in X_target_full.keys()
        }

    elif msca.loss_func == "Gaussian":
        regressor = {
            k: LinearRegression().fit(Z_full[k], X_target_full[k]) for k in Z.keys()
        }

    # Transform sets the cd_rate = 0.0; change back
    msca.cd.cd_rate = 0.5

    # Convert X into a data_loader
    data_loader, _ = convert_to_dataloader(X, batch_size=len(list(X.values())[0]))

    # Repeat for num_bootstraps
    bootstrapped_losses = []
    for _ in tqdm(range(num_bootstraps)):
        # Now iterate through trials in the data_loader
        loss = 0
        for _, (X_target, trial_length) in enumerate(data_loader):
            # Mask the inputs
            X_input_masked, X_output_masked, output_mask, _, Z_r_mask = msca.cd.forward(
                X_target,
                trial_length,
            )

            # Perform a forward pass through the model
            _, Z_r, _ = msca.model(X_input_masked)

            # Compute the masked representations for each region
            Z_r_masked = msca.cd.mask(Z_r, truncate(Z_r_mask, msca.trunc))

            # Reshape inputs + convert to numpy for use with linear regression
            Z_r_masked = {
                k: v.flatten(start_dim=0, end_dim=1).numpy()
                for k, v in Z_r_masked.items()
            }

            # Flatten output mask
            output_mask = {
                k: v[:, msca.trunc].flatten(start_dim=0, end_dim=1).numpy()
                for k, v in output_mask.items()
            }

            # Now make predictions
            predictions = {
                k: torch.tensor(v.predict(Z_r_masked[k])) for k, v in regressor.items()
            }

            # Now mask predictions with output mask
            predictions_masked = msca.cd.mask(predictions, output_mask)

            # Reshape masked output as well
            X_output_masked = {
                k: v[:, msca.trunc].flatten(start_dim=0, end_dim=1)
                for k, v in X_output_masked.items()
            }

            # Correct if needed for Poisson loss
            if msca.loss_func == "Poisson":
                predictions_masked = {
                    k: np.maximum(v, 0) for k, v in predictions_masked.items()
                }

            # Compute the reconstruction loss on the bootstrapped inputs
            loss += sum(
                reconstruction_loss(
                    predictions_masked,
                    X_output_masked,
                    criterion,
                    mode="train",
                )
            )

        # Compute the percent difference in the loss with/without the delay
        bootstrapped_losses.append(loss)

    return np.array(bootstrapped_losses)


def sparsity_sweep_bootstrap_evaluation(
    n_components: int, loss_func: str, X: dict, path: str
):
    """
    This loads in the results from performing a sparsity sweep
    """
    performances = {}
    for sparsity in np.arange(0.0, 1.05, 0.05):
        # Instantiate mSCA with desired sparsity level
        msca = mSCA(
            n_components=n_components,
            n_epochs=1,
            loss_func=loss_func,
            lam_sparse=sparsity.item(),
        )
        msca, losses = msca.fit(X)

        # Perform bootstrap validation
        bootstrapped_losses = bootstrap_performances(msca, X, num_bootstraps=100)

        # Store the distribution
        performances[sparsity] = bootstrapped_losses

    return performances
