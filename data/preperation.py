# from pathlib import Path

# import numpy as np
# import zarr

# from utils import get_tiles, save_tiles, get_segment_paths


# def prepare_pseudo_tile_manifests(data_dir, segments, *, version, tile_size, stride, target, split):
#     data_dir = Path(data_dir)

#     group = "pseudo_labels" if target == "hard" else "prediction"
#     segment_paths = get_segment_paths(data_dir)

#     segments = [str(segment) for segment in segments]
#     for segment in segments:
#         segment_path = segment_paths[segment]
#         output_dir = segment_path / "pseudo_tiles" / str(version) / target

#         npy_path = output_dir / f"tile_size_{tile_size}_stride_{stride}_split_{split if split is not None else "None"}.npy"
#         if npy_path.exists():
#             continue

#         archive = zarr.open(segment_path / f"{segment_path.name}.zarr", mode="r")
#         if group not in archive or str(version) not in archive[group]:
#             raise KeyError(f"Missing pseudo targets '{group}/{version}' for {segment}")
        
#         label = np.asarray(archive[group][str(version)])
#         pseudo_mask = np.asarray(archive["pseudo_masks"][str(version)])
#         mask = np.asarray(archive["mask"]) * pseudo_mask if "mask" in archive else pseudo_mask

#         h, w = archive["volume"].shape[-2:]
#         tiles = get_tiles(label, mask, tile_size, stride, segment, h, w)

#         output_dir.mkdir(parents=True, exist_ok=True)
#         save_tiles(tiles, npy_path, output_dir / f"tile_size_{tile_size}_stride_{stride}_split_{split if split is not None else "None"}.csv")
