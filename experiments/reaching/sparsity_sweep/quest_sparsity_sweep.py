import os
import sys
import shlex
import argparse
import linecache

from msca import *


def get_params_by_id(path_to_file):
    # Get the Array ID from Slurm -> default to 1
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    # Read the specific line from the file
    line = linecache.getline(path_to_file, task_id).strip()

    if not line:
        print(f"Error: Line {task_id} in {path_to_file} is empty/missing.")
        sys.exit(1)

    print(f"Worker #{task_id} processing args: {line}")

    # Split the string into a list (handles quotes correctly)
    return shlex.split(line)


if __name__ == "__main__":
    ### IMPORTANT: ADD PATH TO YOUR params.txt
    experiment_path = "./experiments/reaching/sparsity_sweep/"
    param_path = "params.txt"

    # IMPORTANT: ADD DATA LOADING CODE HERE
    # X, Z_gt, delays_gt = simulate_single_trials(random_seed=0)
    data = torch.load("./experiments/reaching/data/x.pt")
    X = {
        "M1": [x.astype("float32") for x in data["M1"]],
        "PMd": [x.astype("float32") for x in data["PMd"]],
    }

    # Grab parameters passed via cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam_sparse", type=float, required=True)

    parser.add_argument("--n_components", type=int, required=True)

    # Read the specific parameters from the text file
    file_args = get_params_by_id(experiment_path + param_path)

    # Load them in and convert to dictionary
    args = parser.parse_args(file_args)
    args = vars(args)

    # Let's train mSCA with the current values
    msca, losses = mSCA(
        n_components=args["n_components"],
        lam_sparse=args["lam_sparse"],
        n_epochs=7000,
        lam_region=0.0,
        loss_func="Poisson",
        lam_orthog=0.0,
    ).fit(X)

    # Run evaluation
    bootstrapped_losses = bootstrap_performances(msca, X)

    # Save bootstrapped losses
    torch.save(
        bootstrapped_losses,
        f"{experiment_path}bootstrapped_performance_n_components={args['n_components']}_lam_sparse={args['lam_sparse']:.3f}.pt",
    )

    # Save the trained model and corresponding losses
    msca.save(
        f"{experiment_path}msca_n_components={args['n_components']}_lam_sparse={args['lam_sparse']:.3f}.pt"
    )
    torch.save(
        losses,
        f"{experiment_path}losses_n_components={args['n_components']}_lam_sparse={args['lam_sparse']:.3f}.pt",
    )
    print("something")
