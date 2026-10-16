# ink_detection

## Conda

Create conda environmen
```
conda env create -f environment.yml
```

## .env

```
DATA_DIR=<data_dir>
CODE_DIR=<code_dir>
OUTPUT_DIR=<output_dir>
TMP_DIR=<tmp_dir>
```

## Prepare data

Call the following command in data/utils
```
./prepare_data.sh $TMPDIR/data/
```