import torch
import numpy as np
from tqdm import tqdm

from .loss_funcs import *
from .models import *


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
def bootstrap_delays_decoder_old(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 10000
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

    # Compute the predictions on all the data
    X_hat = msca.predict(X, mode="evaluate")

    # Concatenate the predictions across all trials
    X_hat_cat = {k: np.concatenate(v, axis=0) for k, v in X_hat.items()}

    # Concatenate the ground-truth across all trials
    X_cat = {
        k: np.concatenate([t[msca.trunc] for t in v], axis=0) for k, v in X.items()
    }

    # Set the correct evaluation function
    criterion = eval(f"{msca.loss_func}_f".lower())

    # Compute the baseline loss for each region
    baseline_loss = {
        k: criterion(X_hat_cat[k], X_cat[k], mode="evaluate") for k in X_hat_cat.keys()
    }

    # Multiply each region's baseline loss by the region weight
    baseline_loss = {k: v * msca.region_weights[k] for k, v in baseline_loss.items()}

    # Now iteratively remove the learned time-delays and compute the performance change
    performance_changes = {i: [] for i in range(msca.n_components)}
    for i in range(msca.n_components):
        # Zero out the delay for the i-th dimension
        current_delay = msca.model.filters.mus[i].clone()
        msca.model.filters.mus.data[i] = 0

        # Recompute the latent
        X_hat_zeroed = msca.predict(X, mode="evaluate")

        # Concatenate the predictions across all trials
        X_hat_zeroed_cat = {
            k: np.concatenate(v, axis=0) for k, v in X_hat_zeroed.items()
        }

        # Compute the element wise loss for each region
        ew_loss = {
            k: criterion(X_hat_zeroed_cat[k], X_cat[k], mode="evaluate")
            for k in X_hat_cat.keys()
        }

        # Multiply each region's element-wise loss by the region weight
        ew_loss = {k: v * msca.region_weights[k] for k, v in ew_loss.items()}

        # Now compute the difference in the loss without the time-delay
        diff = {k: ew_loss[k] - baseline_loss[k] for k, _ in ew_loss.items()}

        # Concatenate the differences across regions
        diff_concat = np.concatenate([v for _, v in diff.items()], axis=1)

        # Now bootstrap over the differences (only neurons)
        bootstrap_diffs = []
        for _ in tqdm(range(num_bootstraps)):
            # Randomly sample (with replacement) the difference in the loss with/without the delay
            bootstrap_neuron_indices = np.random.choice(
                diff_concat.shape[1], size=diff_concat.shape[1], replace=True
            )

            # Sum over bootstrapped neurons
            bootstrapped_diff = diff_concat[:, bootstrap_neuron_indices].sum()

            # Append bootstrap
            bootstrap_diffs.append(bootstrapped_diff)

        # Reset the time-delay
        msca.model.filters.mus.data[i] = current_delay

        # Store the bootstrap differences
        performance_changes[i] = np.array(bootstrap_diffs)

    return performance_changes


@torch.no_grad()
def bootstrap_delays_decoder(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 1000
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
    data_loader, _ = _convert_to_dataloader(X, shuffle=False)

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
                X_input_masked, X_output_masked, output_mask, Z_mask, Z_r_mask = (
                    msca.cd.forward(
                        X_target,
                        trial_length,
                    )
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
            diffs.append(100 * (without_delay - with_delay) / with_delay)

        performances[i] = np.array(diffs)

    return performances


@torch.no_grad()
def bootstrap_performances(
    msca: object, X: dict[str, np.ndarray], num_bootstraps: int = 1000
) -> dict[int, np.ndarray]:
    """
    This reconstructs the neural activity randomly ablating neurons and
    time-points to induce a distribution over loss values.

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

            # Forward pass with time-delay
            _, _, X_reconstruction = msca.model(X_input_masked)

            # Apply the output mask to the reconstructions
            X_reconstruction_masked = msca.cd.mask(
                X_reconstruction, truncate(output_mask, msca.trunc)
            )

            # Compute the reconstruction loss with the time-delay
            loss += sum(
                reconstruction_loss(
                    X_reconstruction_masked,
                    truncate(X_output_masked, msca.trunc),
                    criterion,
                    mode="train",
                )
            )

        # Compute the percent difference in the loss with/without the delay
        bootstrapped_losses.append(loss.numpy())

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
    for sparsity in np.arange(0.0, 1.05, 0.05):
        # Instantiate mSCA with desired sparsity level
        msca = mSCA(
            n_components=n_components,
            n_epochs=n_epochs,
            loss_func=loss_func,
            lam_sparse=sparsity.item(),
        )
        msca, losses = msca.fit(X)

        # Perform bootstrap validation
        bootstrapped_losses = bootstrap_performances(msca, X)
        torch.save(
            bootstrapped_losses, f"{path}/bootstrapped_sparsity={sparsity.item()}.pt"
        )
        performances[sparsity.item()] = bootstrapped_losses.mean()

        # Save the model
        msca.save(f"{path}/msca_sparsity={sparsity.item()}.pt")

        # Save the losses (for checking convergence)
        torch.save(losses, f"{path}/losses_sparsity={sparsity.item()}.pt")

    # Unpack sparsity values and performances
    sparsity_values = np.array(list(performances.keys()))
    sparsity_performances = np.array(list(performances.values()))

    return sparsity_values[sparsity_performances.argmin()]
