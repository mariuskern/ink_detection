#!/bin/bash -l

#SBATCH --job-name=ink
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
# #SBATCH --mail-type=all
#SBATCH --time=23:59:00 # 08:00:00 23:59:00
#SBATCH --export=NONE
#SBATCH --output=logs/%x_%j.out # logs/<job-name>_<job-id>.out
#SBATCH --error=logs/%x_%j.err # logs/<job-name>_<job-id>.err

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python cuda
conda activate ink

srun python data/utils/staging.py --config-name config

srun python train.py --config-name config
