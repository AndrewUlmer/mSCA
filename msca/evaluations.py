import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from .loss_funcs import *
from .models import *


def pseudo_r2(predictions, X_target):
    # Compute log-likelihood for saturated model
    sat_ll = sum(reconstruction_loss(X_target, X_target, poisson_f, mode="train"))

    # Compute the log-likelihood for the null model
    mean_fr = {k: v.mean(axis=0) for k, v in X_target.items()}
    mean_fr = {
        k: np.stack([mean_fr[k]] * v.shape[0], axis=0) for k, v in X_target.items()
    }
    null_ll = sum(reconstruction_loss(mean_fr, X_target, poisson_f, mode="train"))

    # Compute the actual log-likelihood
    ll = sum(reconstruction_loss(predictions, X_target, poisson_f, mode="train"))

    # Compute the pseudo-r2 - note these are NLLs
    D_model = ll - sat_ll
    D_null = null_ll - sat_ll
    r2 = 1 - (D_model / D_null)

    return r2


def mean_confidence_interval(data: np.ndarray, confidence: float = 0.95):
    """
    Simply computes a one-sides confidence interval based on the bootstrap results
    from bootstrap_delays_decoder

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
                without_delay = sum(
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

    return performances


@torch.no_grad()
def bootstrap_performances(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 1000
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
    """
    # Set the criterion for evaluation
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Convert X into a data_loader
    data_loader, _ = convert_to_dataloader(X)

    # Repeat for num_bootstraps
    bootstrapped_losses = []
    for _ in tqdm(range(num_bootstraps)):
        # Now iterate through trials in the data_loader
        loss = 0
        for _, (X_target, trial_length) in enumerate(data_loader):
            # Apply the mask to the inputs and outputs
            X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                msca.cd.forward(
                    X_target,
                    trial_length,
                )
            )
            # Perform a forward pass through the model
            _, _, X_reconstruction = msca.model(X_input_masked)

            # Mask the reconstruction
            X_reconstruction_masked = msca.cd.mask(
                X_reconstruction, truncate(output_mask, msca.trunc)
            )

            # C = {
            #     k: v / X_reconstruction_masked[k].norm(p=2)
            #     for k, v in truncate(X_output_masked, msca.trunc).items()
            # }

            # Mask the inputs + compute the reconstruction loss
            loss += sum(
                reconstruction_loss(
                    X_reconstruction_masked,
                    truncate(X_output_masked, msca.trunc),
                    criterion,
                    mode="train",
                )
            )

        # Compute the percent difference in the loss with/without the delay
        bootstrapped_losses.append(loss)

    return np.array(bootstrapped_losses)


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
    num_bootstraps: int = 1000,
    threshold: float = 0.01,
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

    ## TESTING: Removing this to see what works for MLP

    # Find relevant latents
    performances = bootstrap_latents_decoder(msca, X, num_bootstraps=100)
    for i in range(msca.n_components):
        mean, lower = mean_confidence_interval(performances[i])
        if lower <= threshold:
            msca.model.decoder_scaling[i] = 0

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
            k: MLPRegressor(
                loss="poisson",
                early_stopping=True,
                max_iter=500,
                activation="logistic",
                random_state=0,
            ).fit(Z_full[k], X_target_full[k])
            for k in Z.keys()
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
            X_input_masked, _, _, _, Z_r_mask = msca.cd.forward(
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

            # Reshape the targets for sklearn regression as well
            X_target = {
                k: v[:, msca.trunc].flatten(start_dim=0, end_dim=1).numpy()
                for k, v in X_target.items()
            }

            # Now make predictions
            predictions = {k: v.predict(Z_r_masked[k]) for k, v in regressor.items()}

            # Correct if needed for Poisson loss
            # if msca.loss_func == "Poisson":
            # predictions = {k: np.maximum(v, 0) for k, v in predictions.items()}
            # loss += pseudo_r2(predictions, X_target)

            # elif msca.loss_func == "Gaussian":

            # Compute the reconstruction loss on the bootstrapped inputs
            loss += sum(
                reconstruction_loss(
                    predictions,
                    X_target,
                    criterion,
                    mode="train",
                )
            )

        # Compute the percent difference in the loss with/without the delay
        bootstrapped_losses.append(loss)

    return np.array(bootstrapped_losses)


def sparsity_sweep_bootstrap(
    n_components: int, n_epochs: int, loss_func: str, X: dict, path: str
):
    """
    This performs a sweep over the sparsity hyperparameter and
    saves the results to a folder specified in path
    """
    print(f"Performing sparsity sweep 🧹 --> saving results to {path}")
    performances = {}

    # Use different ranges for different loss functions
    if loss_func == "Gaussian":
        sparsity_range = np.concatenate(
            [np.array([0.0]), np.array([0.001, 0.01]), np.arange(0.05, 1.05, 0.05)]
        )
    elif loss_func == "Poisson":
        sparsity_range = np.concatenate(
            [
                np.array([0.0]),
                np.array([0.001]),
                np.arange(0.01, 0.1005, 0.005),
            ]
        )

    for sparsity in sparsity_range[3:]:
        # Correcting weird np to python conversion
        sparsity = float(f"{sparsity:0.3f}")

        # Instantiate mSCA with desired sparsity level
        msca = mSCA(
            n_components=n_components,
            n_epochs=n_epochs,
            loss_func=loss_func,
            lam_sparse=1.0,  # sparsity,
            lam_region=0.0,
            lam_orthog=0.0,
        )
        msca, losses = msca.fit(X)

        # Perform bootstrap validation
        bootstrapped_losses = bootstrap_performances_separate_regressor(msca, X)

        # Store the performances
        performances[sparsity] = bootstrapped_losses

        # # Save the losses to confirm the model converged
        # torch.save(
        #     bootstrapped_losses,
        #     f"{path}/bootstrapped_sparsity={sparsity.item():.2f}.pt",
        # )

        # # Save the model so we can retrieve the best model later
        # msca.save(f"{path}/msca_sparsity={sparsity.item():.2f}.pt")

        # # Save the losses (for checking convergence)
        # torch.save(losses, f"{path}/losses_sparsity={sparsity.item():.2f}.pt")

    return performances


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
