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


def get_tiles(label, mask, size, stride, segment_name, h, w, *, positive_threshold=None):
    """Index tiles, measuring positive_fraction over valid pixels only.

    Full tiles are also anchored to the bottom/right edge when needed.
    Without a mask, all pixels are valid. Empty masks yield a positive
    fraction of zero; missing labels yield NaN. Set positive_threshold for
    soft labels to count pixels at or above that probability instead of
    averaging their probabilities. This affects tile statistics only.
    """

    l = []
    # h, w = label.shape[-2:]

    ys = list(range(0, h - size + 1, stride))
    xs = list(range(0, w - size + 1, stride))

    if ys and ys[-1] != h - size:
        ys.append(h - size)
        
    if xs and xs[-1] != w - size:
        xs.append(w - size)

    with tqdm(total=len(ys) * len(xs), desc=f"Tile size {size}, stride {stride}") as pbar:
        for y in ys:
            for x in xs:
                # label_tile = label[y:y+size, x:x+size]
                # mask_tile = mask[y:y+size, x:x+size]

                valid = mask[y:y+size, x:x+size].astype(bool) if mask is not None else None

                if label is not None:
                    label_tile = label[y:y+size, x:x+size]
                    if positive_threshold is not None:
                        label_tile = label_tile >= positive_threshold
                        
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
