#!/bin/bash

set -e

conda activate ink

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

# mkdir -p "$TMPDIR/data_tiff"

# cp -r --verbose /home/vault/b292dc/b292dc23/ink_detection/data/data_tiff/1667 "$TMPDIR/data_tiff/"

mkdir -p "$TMPDIR/data_tiff/0172/unlabeled"

cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250917143559-w062_20250917143559205_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926112011-w078_20250926112011918_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926113336-w079_20250926113336891_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926114310-w080_20250926114310705_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926122938-w082_20250926122938619_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926132117-w083_20250926132117595_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926145427-w084_20250926145427439_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926150345-w086_20250926150345176_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20250926153406-w088_20250926153406704_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251106133152-w077_20251106133152767_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251106170358-w076_20251106170358232_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251107063327-w063_20251107063327215_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251107110950-w064_20251107110950052_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251107175958-w061_20251107175958270_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251109232817-w065_20251109232817724_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251110110112-w066_20251110110112608_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251110135146-w085_20251110135146457_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251110135803-w067_20251110135803677_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251111010954-w068_20251111010954408_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251111151836-w069_20251111151836406_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251111210734-w070_20251111210734722_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251112000002-w087_20251112000002214_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251112163527-w071_20251112163527854_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251113021010-w072_20251113021010674_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251113130119-w073_20251113130119516_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251113152136-w074_20251113152136889_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251113204233-w075_20251113204233755_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251114202035-w081_20251114202035261_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002740-auto_grown_20251115002740308_0_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002741-auto_grown_20251115002740308_1_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002742-auto_grown_20251115002740308_2_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002743-auto_grown_20251115002740308_3_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002744-auto_grown_20251115002740308_4_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002745-auto_grown_20251115002740308_5_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002746-auto_grown_20251115002740308_6_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002747-auto_grown_20251115002740308_7_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251115002748-auto_grown_20251115002740308_8_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251203230500-w089_20251203230500465_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251204060522-w090_20251204060522416_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251204071911-w091_20251204071911070_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251204101314-w092_20251204101314959_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251204192708-w093_20251204192708445_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251205115859-w094_20251205115859448_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251206184729-w095_20251206184729906_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251207123912-w060_20251207123912611_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251210142414-w059_20251210142414060_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251210153304-w058_20251210153304503_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251210160810-w057_20251210160810301_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251211095724-w056_20251211095724922_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251211111451-w055_20251211111451620_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251211135303-w054_20251211135303394_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251211152543-w053_20251211152543204_flatboi "$TMPDIR/data_tiff/0172/unlabeled"
# cp -r --verbose /home/atuin/b292dc/b292dc23/ink_detection/data/data_fei/new-inference-surface-volumes/20251212105627-w052_20251212105627148_flatboi "$TMPDIR/data_tiff/0172/unlabeled"

echo

# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------

echo "=== Running prepare_data.sh ==="

cd /home/vault/b292dc/b292dc23/ink_detection/code/data/utils

./prepare_data.sh "$TMPDIR" --workers 4
# ./prepare_data.sh "$TMPDIR" --segments w018_20240304144031_2um --crops w018_20240304144031_2um:0,42632,0,23000 w018_20240304144031_2um:0,42632,23000,54000 w018_20240304144031_2um:0,42632,54000,84000 w018_20240304144031_2um:0,42632,84000,98362 --workers 2

# ------------------------------------------------------------
# 3. Ergebnis zurückkopieren
# ------------------------------------------------------------

echo
echo "=== Copying results ==="

# cp -r --verbose "$TMPDIR/data_tar" /home/atuin/b292dc/b292dc23/ink_detection/
cp -r --verbose "$TMPDIR/data_tar" /home/atuin/b292dc/b292dc23/ink_detection/data_tar_fei

echo
echo "=== DONE ==="
echo "End: $(date)"

exit 0