from pathlib import Path

import zarr
import numpy as np
import tifffile


z = zarr.open(Path("./output/20241025145341/20241025145341.zarr"), mode="r")
z = np.array(z["volume"])

d, h, w = z.shape

files = sorted(Path("D:/Dokumente/Studium/Masterarbeit/omar/data_fei/new_labeled/20241025145341/layers/").glob("*.tif"))
t = np.stack([tifffile.imread(f) for f in files])

print(np.array_equal(z, t[:d, :h, :w]))