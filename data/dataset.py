from pathlib import Path
import json

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class InkDataset(Dataset):
    """
    PyTorch Dataset for loading image tiles from multiple segments stored in Zarr archives.

    Each segment is stored in a separate directory containing:
    - a Zarr archive with the image volume (and optionally labels)
    - a NumPy tile index file containing tile coordinates and metadata

    The dataset uses the tile index to retrieve individual tiles on demand
    from the corresponding Zarr archive, avoiding loading the complete volume
    into memory.

    Parameters
    ----------
    data_dir : str or Path
        Root directory containing segment directories.

    segments : list[str], optional
        Names of the segments to include. If None, all available segments
        are loaded.

    tile_size : int, optional
        Spatial size of each extracted tile (height and width).

    stride : int, optional
        Stride used during tile generation. If None, defaults to half of
        the tile size.

    transform : callable, optional
        Optional transformation applied to each image tile before returning it.

    Returns
    -------
    tuple
        A tuple containing:
        - tile : torch.Tensor
            Image tile of shape (tile_size, tile_size).
        - label : torch.Tensor or None
            Corresponding label tile if available, otherwise None.
        - segment : str
            Name of the source segment.
        - x : int
            X-coordinate of the tile within the original volume.
        - y : int
            Y-coordinate of the tile within the original volume.
    """

    def __init__(
        self,
        data_dir,
        segments=None,
        tile_size=64,
        stride=None,
        n_layers=65,
        center=None,
        depth=32,
        norm="global",
        min_mask_fraction=0.5,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size // 2
        self.n_layers = n_layers
        self.center = center if center is not None else n_layers // 2
        self.depth = depth
        self.norm = norm
        self.transform = transform

        if self.norm not in ["global", "local"]:
            raise ValueError(
                f"Invalid normalization mode: {self.norm}. Must be 'global' or 'local'."
            )

        self.segments_to_paths = {}
        for folder in self.data_dir.iterdir():
            for segment in folder.iterdir():
                if segment.is_dir():
                    self.segments_to_paths[segment.name] = segment

        if segments is None:
            # self.segments = sorted([p.name for p in self.data_dir.iterdir() if p.is_dir()])
            self.segments = sorted(self.segments_to_paths.keys())
        else:
            self.segments = list(map(str, segments))

        self.tiles = []
        self.zarr_data = {}
        self.segment_shapes = {}

        for segment in self.segments:
            # Load tiles

            tiles_path = (
                self.segments_to_paths[segment]
                / "tiles"
                / f"tile_size_{self.tile_size}_stride_{self.stride}.npy"
            )

            if not tiles_path.exists():
                raise FileNotFoundError(f"Missing tile index: {tiles_path}")

            tiles = np.load(tiles_path, allow_pickle=False)

            for entry in tiles:
                if entry["mask_fraction"] < min_mask_fraction:
                    continue
                
                self.tiles.append(
                    {
                        "segment": segment,
                        "y": int(entry["y"]),
                        "x": int(entry["x"]),
                        "positive_fraction": float(entry["positive_fraction"]),
                        "mask_fraction": float(entry["mask_fraction"]),
                    }
                )

            # Load zarr

            zarr_path = self.segments_to_paths[segment] / f"{segment}.zarr"

            if not zarr_path.exists():
                raise FileNotFoundError(f"Missing zarr archive: {zarr_path}")

            self.zarr_data[segment] = zarr.open(zarr_path, mode="r")

            # Load meta

            meta_path = self.segments_to_paths[segment] / "metadata.json"

            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata file: {meta_path}")

            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.segment_shapes[segment] = meta["volume_shape"]

        self.z0 = self.center - self.depth // 2
        self.z1 = self.z0 + self.depth

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        entry = self.tiles[idx]

        segment = entry["segment"]
        x = entry["x"]
        y = entry["y"]

        zarr_array = self.zarr_data[segment]

        tile = torch.from_numpy(
            zarr_array["volume"][
                self.z0 : self.z1, y : y + self.tile_size, x : x + self.tile_size
            ]
        ).float()
        if "label" in zarr_array:
            label = torch.from_numpy(
                zarr_array["label"][y : y + self.tile_size, x : x + self.tile_size]
            ).float()
        else:
            label = None

        if self.transform:
            tile = self.transform(tile)

        tile = self._normalize(tile, mode=self.norm)

        tile = tile.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None

        return tile, label, segment, x, y, torch.tensor(self.segment_shapes[segment])

    def _normalize(self, tile, mode="global"):
        match mode:
            case "global":
                return tile / 255.0
            case "local":
                z = (tile - tile.mean()) / (tile.std() + 1e-8)
                z = np.clip(z, -3, 3)
                z = (z + 3) / 6
                return z
