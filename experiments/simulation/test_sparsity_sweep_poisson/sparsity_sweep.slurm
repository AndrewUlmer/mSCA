#!/bin/bash
#SBATCH --job-name=latent_sparsity_sweep
#SBATCH --account=p31796
#SBATCH --partition=short
#SBATCH --output=./quest/logs/job_%A_%a.out
#SBATCH --nodes=1                 # 1 node per hyperparameter combo
#SBATCH --ntasks-per-node=10	  # Total requested RAM will be 10 x 10 = 100G
#SBATCH --mem-per-cpu=10G         # Adjust as needed
#SBATCH --time=4:00:00            # Time required to run
#SBATCH --array=1-21               # Array range must match lines in params.txt

# Get the specific parameters for this array task
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./experiments/simulation/test_sparsity_sweep_poisson/params.txt)

echo "Running job ${SLURM_ARRAY_TASK_ID} with params: $PARAMS"

# Run conda
source activate
conda activate mSCA

# Run your script
python ./experiments/simulation/test_sparsity_sweep_poisson/quest_sparsity_sweep.py $PARAMS
