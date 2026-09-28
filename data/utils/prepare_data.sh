#!/bin/bash

set -e  # Abbruch bei Fehlern

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_folder> [options]"
    exit 1
fi

BASE_DIR=$1
shift

TIFF_DIR="${BASE_DIR}/data_tiff"
ZARR_DIR="${BASE_DIR}/data_zarr"
TAR_DIR="${BASE_DIR}/data_tar"


# Optionale Parameter
SEGMENTS=""
CHUNKS=""
TILE_SIZES=""
STRIDES=""


# Argumente parsen
while [[ $# -gt 0 ]]; do
    case $1 in

        --segments)
            shift
            SEGMENTS="--segments"
            while [[ $# -gt 0 && $1 != --* ]]; do
                SEGMENTS+=" $1"
                shift
            done
            ;;

        --chunks)
            CHUNKS="--chunks $2 $3 $4"
            shift 4
            ;;

        --crops)
            shift
            CROPS="--crops"
            while [[ $# -gt 0 && $1 != --* ]]; do
                CROPS+=" $1"
                shift
            done
            ;;

        # --tile-sizes)
        #     shift
        #     TILE_SIZES="--tile-sizes"
        #     while [[ $# -gt 0 && $1 != --* ]]; do
        #         TILE_SIZES+=" $1"
        #         shift
        #     done
        #     ;;

        # --strides)
        #     shift
        #     STRIDES="--strides"
        #     while [[ $# -gt 0 && $1 != --* ]]; do
        #         STRIDES+=" $1"
        #         shift
        #     done
        #     ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done


# echo "=============================="
echo "Input:"
echo "  TIFF: $TIFF_DIR"
echo "  ZARR: $ZARR_DIR"
echo "  TAR:  $TAR_DIR"
echo
echo "=============================="


echo
echo "Running TIFF -> Zarr conversion..."
echo

python -u tiff_to_zarr.py \
    "$TIFF_DIR" \
    "$ZARR_DIR" \
    $SEGMENTS \
    $CHUNKS \
    $CROPS

echo "=============================="


# echo
# echo "Generating tiles..."
# echo

# python zarr_to_tiles.py \
#     "$ZARR_DIR" \
#     $TILE_SIZES \
#     $STRIDES

# echo "=============================="


echo
echo "Creating tar archives..."
echo

python -u create_tar.py \
    "$ZARR_DIR" \
    "$TAR_DIR"


echo
echo "Pipeline finished successfully."