from pathlib import Path
import json

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, ConcatDataset
from torchvision import tv_tensors

from utils import get_tiles, save_tiles


def collate_fn(batch):
    result = {}

    result["tile"] = torch.stack([b["tile"] for b in batch])

    for key in ["label", "mask"]:
        values = [b[key] for b in batch]
        result[key] = (torch.stack(values) if values[0] is not None else None)

    result["is_pseudo"] = torch.tensor([b["is_pseudo"] for b in batch])

    result["segment"] = [b["segment"] for b in batch]
    result["x"] = [b["x"] for b in batch]
    result["y"] = [b["y"] for b in batch]
    result["shape"] = torch.stack([b["shape"] for b in batch])

    return result


class CombinedInkDataset(ConcatDataset):
    @property
    def tiles(self):
        return [tile for dataset in self.datasets for tile in dataset.tiles]


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
        use_pseudo_labels=False,
        pseudo_label_version=None,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size // 2
        self.n_layers = n_layers
        self.center = center if center is not None else n_layers // 2
        self.depth = depth
        self.norm = norm
        self.use_pseudo_labels = use_pseudo_labels
        self.pseudo_label_version = pseudo_label_version
        self.transform = transform

        if self.norm not in ["global", "local"]:
            raise ValueError(f"Invalid normalization mode: {self.norm}. Must be 'global' or 'local'.")

        segments_to_paths = {}
        for folder in self.data_dir.iterdir():
            for segment in folder.iterdir():
                if segment.is_dir():
                    segments_to_paths[segment.name] = segment

        if segments is None:
            self.segments = sorted(segments_to_paths.keys())
        else:
            self.segments = list(map(str, segments))

        self.tiles = []
        self.zarr_data = {}
        self.segment_shapes = {}

        for segment in self.segments:
            # Load zarr data
            self.zarr_data[segment] = self._get_zarr_data_for_segment(segments_to_paths[segment])

            # Load tiles
            self.tiles.extend(self._get_tiles_for_segment(segments_to_paths[segment], min_mask_fraction))

            # Load meta
            self.segment_shapes[segment] = torch.tensor(self._get_meta_for_segment(segments_to_paths[segment])["volume_shape"])

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
        
        tile = self._get_tile(zarr_array, x, y)
        label = self._get_label(zarr_array, x, y)
        mask = self._get_mask(zarr_array, x, y)

        is_pseudo = self.use_pseudo_labels

        tile = tv_tensors.Image(tile)
        label = tv_tensors.Mask(label) if label is not None else None
        mask = tv_tensors.Mask(mask) if mask is not None else None

        if self.transform is not None:
            tile, label, mask = self.transform(tile, label, mask)

        tile = tile.unsqueeze(0)
        label = label.unsqueeze(0) if label is not None else None
        mask = mask.unsqueeze(0) if mask is not None else None

        # return tile, label, mask, is_pseudo, segment, x, y, self.segment_shapes[segment]
        return {
            "tile": tile,
            "label": label,
            "mask": mask,
            "is_pseudo": is_pseudo,
            "segment": segment,
            "x": x,
            "y": y,
            "shape": self.segment_shapes[segment],
        }

    def _normalize(self, tile, mode="global"):
        match mode:
            case "global":
                return tile / 255.0
            case "local":
                z = (tile - tile.mean()) / (tile.std() + 1e-8)
                z = torch.clamp(z, -3, 3) # z = np.clip(z, -3, 3)
                z = (z + 3) / 6
                return z

    def _get_tile(self, zarr_array, x, y):
        tile = torch.from_numpy(zarr_array["volume"][self.z0:self.z1, y:y+self.tile_size, x:x+self.tile_size]).float()

        tile = self._normalize(tile, mode=self.norm)

        # if self.transform:
        #     tile = self.transform(tile)

        return tile

    def _get_label(self, zarr_array, x, y):
        if self.use_pseudo_labels:
            version = str(self.pseudo_label_version)
            label = torch.from_numpy(zarr_array["pseudo_labels"][version][y:y+self.tile_size, x:x+self.tile_size]).float()
        elif "label" in zarr_array:
            label = torch.from_numpy(zarr_array["label"][y:y+self.tile_size, x:x+self.tile_size]).float()
        else:
            label = None

        return label

    def _get_mask(self, zarr_array, x, y):
        if self.use_pseudo_labels:
            version = str(self.pseudo_label_version)
            pseudo_mask = torch.from_numpy(zarr_array["pseudo_masks"][version][y:y+self.tile_size, x:x+self.tile_size]).float()
            if "mask" in zarr_array:
                mask = torch.from_numpy(zarr_array["mask"][y:y+self.tile_size, x:x+self.tile_size]).float()
                mask = mask * pseudo_mask
            else:
                mask = pseudo_mask
        elif "mask" in zarr_array:
            mask = torch.from_numpy(zarr_array["mask"][y:y+self.tile_size, x:x+self.tile_size]).float()
        else:
            mask = None

        return mask

    def _get_tiles_for_segment(self, segment_path, min_mask_fraction):
        if self.use_pseudo_labels:
            tile_path = segment_path / "pseudo_tiles" / str(self.pseudo_label_version) / f"tile_size_{self.tile_size}_stride_{self.stride}.npy"
            csv_path = segment_path / "pseudo_tiles" / str(self.pseudo_label_version) / f"tile_size_{self.tile_size}_stride_{self.stride}.csv"
        else:
            tile_path = segment_path / "tiles" / f"tile_size_{self.tile_size}_stride_{self.stride}.npy"
            csv_path = segment_path / "tiles" / f"tile_size_{self.tile_size}_stride_{self.stride}.csv"
        
        # if not tile_path.exists():
        #     raise FileNotFoundError(f"Missing tile index: {tile_path}")

        if tile_path.exists():
            tiles = np.load(tile_path, allow_pickle=False)
        else:
            print(f"Missing tile index: {tile_path}. Creating tiles.")

            zarr_data = self.zarr_data[segment_path.name]
            volume_shape = zarr_data["volume"].shape
            h, w = volume_shape[-2:]

            # if self.use_pseudo_labels:
            #     version = str(self.pseudo_label_version)
            #     label = np.asarray(zarr_data["pseudo_labels"][version])
            #     mask = np.asarray(zarr_data["pseudo_masks"][version])
            # else:
            #     label = np.asarray(zarr_data["label"]) if "label" in zarr_data else None
            #     mask = np.asarray(zarr_data["mask"]) if "mask" in zarr_data else None

            if self.use_pseudo_labels:
                version = str(self.pseudo_label_version)
                label = np.asarray(zarr_data["pseudo_labels"][version])
                pseudo_mask = np.asarray(zarr_data["pseudo_masks"][version])

                if "mask" in zarr_data:
                    true_mask = np.asarray(zarr_data["mask"])
                    mask = true_mask * pseudo_mask
                else:
                    mask = pseudo_mask
            else:
                label = np.asarray(zarr_data["label"]) if "label" in zarr_data else None
                mask = np.asarray(zarr_data["mask"]) if "mask" in zarr_data else None

            tiles = get_tiles(
                label,
                mask,
                self.tile_size,
                self.stride,
                segment_path.name,
                h,
                w
            )

            tile_path.parent.mkdir(parents=True, exist_ok=True)
            save_tiles(tiles, tile_path, csv_path)

        tile_list = []

        for t in tiles:
            if t["mask_fraction"] < min_mask_fraction:
                continue
            
            tile_list.append(
                {
                    "segment": segment_path.name,
                    "y": int(t["y"]),
                    "x": int(t["x"]),
                    "positive_fraction": float(t["positive_fraction"]),
                    "mask_fraction": float(t["mask_fraction"]),
                }
            )

        return tile_list

    def _get_zarr_data_for_segment(self, segment_path):
        zarr_path = segment_path / f"{segment_path.name}.zarr"
        
        if not zarr_path.exists():
            raise FileNotFoundError(f"Missing zarr archive: {zarr_path}")

        zarr_data = zarr.open(zarr_path, mode="r")

        if self.use_pseudo_labels:
            if "pseudo_labels" not in zarr_data:
                raise KeyError(f"Missing pseudo labels in {zarr_path}.")

            pseudo_labels = zarr_data["pseudo_labels"]
            available_versions = list(pseudo_labels.keys())

            if self.pseudo_label_version is None:
                self.pseudo_label_version = int(available_versions[-1])
            else:
                version = str(self.pseudo_label_version)
                if version not in pseudo_labels:
                    raise KeyError(f"Missing pseudo label version '{version}'. Available versions: {available_versions}")

            # if self.pseudo_label_version is None:
            #     raise ValueError("pseudo_label_version must be specified when use_pseudo_labels=True")

            # version = str(self.pseudo_label_version)

            # if version not in pseudo_labels:
            #     raise KeyError(f"Missing pseudo label version '{version}'. Available versions: {available_versions}")

        return zarr_data

    def _get_meta_for_segment(self, segment_path):
        meta_path = segment_path / "metadata.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return meta
