#!/bin/bash

set -e

conda activate ink

# ------------------------------------------------------------
# 1. Daten in den lokalen TMPDIR kopieren
# ------------------------------------------------------------

echo "=== Preparing TMPDIR ==="

mkdir -p "$TMPDIR/data_tiff/1667/labeled"

# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w013_20240304141531_2um "$TMPDIR/data_tiff/1667/labeled/"
# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w018_20240304144031_2um "$TMPDIR/data_tiff/1667/labeled/"
cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w028_20251208130119156_2um "$TMPDIR/data_tiff/1667/labeled/"
cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w029_20251212185248662_2um "$TMPDIR/data_tiff/1667/labeled/"
cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667/labeled/w031_2025122323_2um "$TMPDIR/data_tiff/1667/labeled/"

# mkdir -p "$TMPDIR/data_tiff"

# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667 "$TMPDIR/data_tiff/"

echo

# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------

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

exit 0