#!/bin/bash -l

#SBATCH --job-name=ink_array
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --export=NONE
#SBATCH --array=0-4%5
#SBATCH --output=logs/%x_%j.out # logs/<job-name>_<job-id>.out
#SBATCH --error=logs/%x_%j.err # logs/<job-name>_<job-id>.err

# Run from the repository root:
# mkdir -p logs
# sbatch slurm_job_scripts/train_array.sbatch a1
# Each task starts the selected config normally, overriding only train.seed.
set -eo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch slurm_job_scripts/train_array.sbatch <config-name>" >&2
    exit 2
fi

: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch as an array.}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 4 )); then
    echo "Expected an array task index between 0 and 4." >&2
    exit 2
fi

config_name=$1
# Keep the spacing at least as large as train.iterations to avoid overlap.
seed=$((42 + 1000 * SLURM_ARRAY_TASK_ID))

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python cuda
conda activate ink

echo "Config: ${config_name} | Seed: ${seed} | Array task: ${SLURM_ARRAY_TASK_ID}"

srun python data/utils/staging.py --config-name "$config_name" "train.seed=$seed"
srun python train.py --config-name "$config_name" "train.seed=$seed"
