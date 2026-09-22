#!/bin/bash -l

#SBATCH --job-name=preprocess
#SBATCH --ntasks=1
#SBATCH --partition=a100mig
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=128G
# #SBATCH --mail-type=all
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=logs/preprocess/%x_%j.out
#SBATCH --error=logs/preprocess/%x_%j.err

unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python cuda
conda activate ink

echo "=== JOB INFO ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "TMPDIR:       $TMPDIR"
echo "Start:        $(date)"
echo

# ------------------------------------------------------------
# 1. Daten in den lokalen TMPDIR kopieren
# ------------------------------------------------------------

echo "=== Preparing TMPDIR ==="

# mkdir -p "$TMPDIR/data_tiff/1667/labeled"

# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w013_20240304141531_2um "$TMPDIR/data_tiff/1667/labeled/"
# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w018_20240304144031_2um "$TMPDIR/data_tiff/1667/labeled/"
# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w028_20251208130119156_2um "$TMPDIR/data_tiff/1667/labeled/"
# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w029_20251212185248662_2um "$TMPDIR/data_tiff/1667/labeled/"
# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w031_2025122323_2um "$TMPDIR/data_tiff/1667/labeled/"

mkdir -p "$TMPDIR/data_tiff"

cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667 "$TMPDIR/data_tiff/"

echo

# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------

echo "=== RESOURCES ==="
free -h
echo
scontrol show job "$SLURM_JOB_ID" | grep -E "NumCPUs|MinMemory|TRES|AllocTRES"

echo "=== Running prepare_data.sh ==="

cd /home/vault/b292dc/b292dc23/ink_detection/code/data/utils

./prepare_data.sh "$TMPDIR"

# ------------------------------------------------------------
# 3. Ergebnis zurückkopieren
# ------------------------------------------------------------

echo
echo "=== Copying results ==="

cp -r --verbose "$TMPDIR/data_tar" /home/atuin/b292dc/b292dc23/ink_detection/

echo
echo "=== DONE ==="
echo "End: $(date)"