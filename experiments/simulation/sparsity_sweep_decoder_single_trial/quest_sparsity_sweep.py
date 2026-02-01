import os
from msca import *

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
    # Set experiment path to load parameters from
    experiment_path = "./experiments/simulation/sparsity_sweep_decoder_single_trial/"
    param_path = "params.txt"

    # Load in the dataset (simulated)
    if not os.path.exists("./experiments/simulation/data/x.pt"):
        X, Z_gt, delays_gt = simulate_single_trials(random_seed=0)
        torch.save(X, f"./experiments/simulation/data/x.pt")
    else:
        X = torch.load(f"./experiments/simulation/data/x.pt")

    # Grab parameters passed via cli
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam_sparse", type=float, required=True)

    # Read the specific parameters from the text file
    file_args = get_params_by_id(experiment_path + param_path)

    # Load them in and convert to dictionary
    args = parser.parse_args(file_args)
    args = vars(args)

    # This will save pre/post scaling
    msca, losses = mSCA(
        n_components=5,
        n_epochs=8000,
        loss_func="Poisson",
        lam_sparse=args["lam_sparse"],
    ).fit(X)
