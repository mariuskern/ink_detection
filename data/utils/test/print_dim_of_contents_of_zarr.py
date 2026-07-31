from pathlib import Path

import numpy as np
import zarr


PATH = Path("./output/")

z = zarr.open(PATH / "20241025145341.zarr", mode="r")

print(z["volume"].shape)
print(z["label"].shape)
print(z["mask"].shape)