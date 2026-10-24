import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm


TILE_DTYPE = np.dtype(
    [
        ("segment", "U128"),
        ("y", np.int32),
        ("x", np.int32),
        ("positive_fraction", np.float32),
        ("mask_fraction", np.float32),
    ]
)


def get_tiles(label, mask, size, stride, segment_name, h, w):
    """Index tiles, measuring positive_fraction over valid pixels only.

    Without a mask, all pixels are valid. Empty masks yield a positive
    fraction of zero; missing labels yield NaN.
    """

    l = []
    # h, w = label.shape[-2:]

    with tqdm(total=((h - size) // stride + 1) * ((w - size) // stride + 1), desc=f"Tile size {size}, stride {stride}") as pbar:
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                # label_tile = label[y:y+size, x:x+size]
                # mask_tile = mask[y:y+size, x:x+size]

                valid = mask[y:y+size, x:x+size].astype(bool) if mask is not None else None

                if label is not None:
                    label_tile = label[y:y+size, x:x+size]
                    if valid is None:
                        positive_fraction = float(label_tile.mean())
                    else:
                        positive_fraction = float(label_tile[valid].mean()) if valid.any() else 0.0
                else:
                    positive_fraction = np.nan

                if mask is not None:
                    mask_tile = mask[y:y+size, x:x+size]
                    mask_fraction = float(mask_tile.mean())
                else:
                    mask_fraction = np.nan

                l.append(
                    (
                        segment_name,
                        y,
                        x,
                        positive_fraction,
                        mask_fraction,
                    )
                )

                pbar.update(1)

    l = np.asarray(l, dtype=TILE_DTYPE)

    return l

def save_tiles(tiles, npy_path, csv_path):
    np.save(npy_path, tiles)
    
    np.savetxt(
        csv_path,
        tiles,
        delimiter=",",
        header=",".join(tiles.dtype.names),
        comments="",
        fmt=[
            "%s",
            "%d",
            "%d",
            "%.6f",
            "%.6f",
        ],
    )
