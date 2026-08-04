import os
from pathlib import Path
from PIL import Image

import zarr
import numpy as np
import tifffile

from dotenv import load_dotenv
load_dotenv()


z = zarr.open(Path(os.getenv("DATA_DIR")) / "labeled" / "20241025145341/20241025145341.zarr", mode="r")
volume = np.array(z["volume"])

d, h, w = volume.shape


# Volume

files = sorted(Path("D:/Dokumente/Studium/Masterarbeit/omar/data_fei/new_labeled/20241025145341/layers/").glob("*.tif"))
t = np.stack([tifffile.imread(f) for f in files])

print(np.array_equal(volume, t[:d, :h, :w]))


# Label

label_path = Path("D:/Dokumente/Studium/Masterarbeit/omar/data_fei/new_labeled/20241025145341/") / "20241025145341_inklabels.png"
if label_path.exists():
    label = np.array(Image.open(label_path).convert("L"))
    label = (label > 127).astype(np.uint8)
    z_label = np.array(z["label"])
    print(np.array_equal(z_label, label[:h, :w]))


# Maks

mask_path = Path("D:/Dokumente/Studium/Masterarbeit/omar/data_fei/new_labeled/20241025145341/") / "20241025145341_mask.png"
if mask_path.exists():
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype(np.uint8)
    z_mask = np.array(z["mask"])
    print(np.array_equal(z_mask, mask[:h, :w]))
