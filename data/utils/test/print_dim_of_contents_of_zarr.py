from pathlib import Path

import zarr


z = zarr.open(Path("./output/20241025145341/") / "20241025145341.zarr", mode="r")

print(z["volume"].shape)
print(z["label"].shape)
print(z["mask"].shape)