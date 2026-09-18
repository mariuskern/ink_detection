from pathlib import Path
import json

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, ConcatDataset
from torchvision import tv_tensors

from utils import get_tiles, save_tiles, get_segment_paths


class CombinedInkDataset(ConcatDataset):
    @property
    def tiles(self):
        return [tile for dataset in self.datasets for tile in dataset.tiles]


class InkDataset(Dataset):
    """
    PyTorch Dataset for loading image tiles from Zarr-based Ink data.

    Each segment is stored in a separate directory containing:
    - a Zarr archive with the image volume
    - optionally a ground-truth label
    - optionally a ground-truth validity mask
    - optionally versioned pseudo labels
    - optionally versioned pseudo-label validity/confidence masks
    - a tile index used to determine valid tiles

    Tiles are extracted from the volume using the configured tile size,
    stride, center slice and depth. The image volume is normalized either
    globally to [0, 1] or locally using per-tile standardization.

    If `pseudo_label_version` is specified, pseudo targets and their
    corresponding masks are loaded from the selected version. Depending on
    `pseudo_target`, either hard pseudo labels (`pseudo_labels`) or soft
    predictions (`prediction`) are used.

    The dataset always returns tensors for label, pseudo_label, mask and
    pseudo_mask. If one of these targets is not available, a dummy tensor
    is returned together with a corresponding boolean availability flag.

    Returned dictionary
    -------------------
    tile:
        Image tile of shape [1, D, H, W], where D is the configured
        depth of the input volume.

    label:
        Ground-truth segmentation label of shape [1, H, W].
        If no ground-truth label is available, a zero tensor is returned.

    pseudo_label:
        Versioned pseudo segmentation target of shape [1, H, W].
        For `pseudo_target="hard"`, this is loaded from `pseudo_labels`.
        For `pseudo_target="soft"`, this is loaded from `prediction`.
        If no pseudo label is available, a zero tensor is returned.

    mask:
        Ground-truth validity mask of shape [1, H, W].
        If no ground-truth mask is available, a tensor of ones is returned.

    pseudo_mask:
        Versioned pseudo-label validity/confidence mask of shape [1, H, W].
        If no pseudo mask is available, a tensor of ones is returned.

    has_label:
        Boolean tensor indicating whether a ground-truth label exists.

    has_pseudo_label:
        Boolean tensor indicating whether the requested pseudo target exists.

    has_mask:
        Boolean tensor indicating whether a ground-truth validity mask exists.

    has_pseudo_mask:
        Boolean tensor indicating whether the requested pseudo-label mask exists.

    segment:
        Name of the source segment.

    x, y:
        Top-left tile coordinates in the original volume.

    shape:
        Original spatial volume shape of the source segment.

    Parameters
    ----------
    data_dir:
        Root directory containing the segment directories.

    segments:
        Optional iterable specifying the segments to include. If `None`,
        all available segments are used.

    tile_size:
        Spatial size of each square tile.

    stride:
        Spatial stride between neighboring tiles. Defaults to
        `tile_size // 2`.

    n_layers:
        Number of available input layers around the selected center.

    center:
        Center layer of the input volume. Defaults to the middle layer.

    depth:
        Number of consecutive layers used as input for each tile.

    norm:
        Image normalization mode. Either `"global"` or `"local"`.

    min_mask_fraction:
        Minimum fraction of valid pixels required for a tile to be included.

    pseudo_label_version:
        Version of the pseudo targets to load. If `None`, no pseudo targets
        are used.

    pseudo_target:
        Type of pseudo target to use. `"hard"` loads binary pseudo labels
        from `pseudo_labels`, while `"soft"` loads continuous predictions
        from `prediction`.

    transform:
        Optional transformation applied jointly to the image, labels and
        masks.
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
        use_as_unlabeled=False,
        pseudo_label_version=None,
        pseudo_target="hard",
        transform=None,
        split=None
    ):
        self.data_dir = Path(data_dir)
        self.tile_size = tile_size
        self.stride = stride if stride is not None else tile_size // 2
        self.n_layers = n_layers
        self.center = center if center is not None else n_layers // 2
        self.depth = depth
        self.norm = norm
        self.use_as_unlabeled = use_as_unlabeled
        self.pseudo_label_version = pseudo_label_version
        self.pseudo_target = pseudo_target
        self.transform = transform
        self.split = split

        if self.norm not in ["global", "local"]:
            raise ValueError(f"Invalid normalization mode: {self.norm}. Must be 'global' or 'local'.")
        
        if self.pseudo_target not in ["hard", "soft"]:
            raise ValueError(f"Invalid pseudo target '{self.pseudo_target}'. Must be 'hard' or 'soft'.")

        if split is not None and split not in ["train", "val"]:
            raise ValueError(f"Invalid split '{self.split}'. Must be None, 'train' or 'val'.")

        self.segment_paths = get_segment_paths(self.data_dir)

        if segments is None:
            self.segments = sorted(self.segment_paths.keys())
        else:
            self.segments = list(map(str, segments))

        self.tiles = []
        self.zarr_data = {}
        self.segment_shapes = {}

        for segment in self.segments:
            if segment not in self.segment_paths:
                raise KeyError(f"Unknown segment '{segment}'. Available segments: {sorted(self.segment_paths.keys())}")

            segment_path = self.segment_paths[segment]
            
            # Load zarr data
            self.zarr_data[segment] = self._get_zarr_data_for_segment(segment_path)

            # Load tiles
            self.tiles.extend(self._get_tiles_for_segment(segment, min_mask_fraction))

            # Load meta
            self.segment_shapes[segment] = torch.tensor(self._get_meta_for_segment(segment_path)["volume_shape"])

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
        pseudo_label = self._get_pseudo_label(zarr_array, x, y)
        mask = self._get_mask(zarr_array, x, y)
        pseudo_mask = self._get_pseudo_mask(zarr_array, x, y)
        
        if label is not None:
            self._assert_binary(label, "label")

        if pseudo_label is not None and self.pseudo_target == "hard":
            self._assert_binary(pseudo_label, "pseudo_label")

        if mask is not None:
            self._assert_binary(mask, "mask")

        if pseudo_mask is not None:
            self._assert_binary(pseudo_mask, "pseudo_mask")

        has_label = label is not None
        has_pseudo_label = pseudo_label is not None
        has_mask = mask is not None
        has_pseudo_mask = pseudo_mask is not None

        # Dummy-Tensoren erzeugen
        if label is None:
            label = torch.zeros((self.tile_size, self.tile_size), dtype=torch.float32)
        if pseudo_label is None:
            pseudo_label = torch.zeros_like(label)
        if mask is None:
            mask = torch.ones_like(label)
        if pseudo_mask is None:
            pseudo_mask = torch.ones_like(label)

        tile = tv_tensors.Image(tile)
        label = tv_tensors.Mask(label)
        pseudo_label = tv_tensors.Mask(pseudo_label)
        mask = tv_tensors.Mask(mask)
        pseudo_mask = tv_tensors.Mask(pseudo_mask)

        if self.transform is not None:
            tile, label, pseudo_label, mask, pseudo_mask = self.transform(tile, label, pseudo_label, mask, pseudo_mask)

        tile = tile.unsqueeze(0) # [1, D, H, W]
        label = label.unsqueeze(0) # [1, H, W]
        pseudo_label = pseudo_label.unsqueeze(0)
        mask = mask.unsqueeze(0)
        pseudo_mask = pseudo_mask.unsqueeze(0)

        return {
            "tile": tile,

            "label": label,
            "pseudo_label": pseudo_label,
            "mask": mask,
            "pseudo_mask": pseudo_mask,

            "has_label": torch.tensor(has_label, dtype=torch.bool),
            "has_pseudo_label": torch.tensor(has_pseudo_label,dtype=torch.bool),
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
            "has_pseudo_mask": torch.tensor(has_pseudo_mask, dtype=torch.bool),

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
        return tile

    def _get_label(self, zarr_array, x, y):
        if "label" not in zarr_array or self.use_as_unlabeled:
            return None

        label = torch.from_numpy(zarr_array["label"][y:y+self.tile_size, x:x+self.tile_size]).float()
        
        return label

    def _get_pseudo_label(self, zarr_array, x, y):
        if self.pseudo_label_version is None:
            return None

        group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
        if group not in zarr_array:
            return None

        version = str(self.pseudo_label_version)
        if version not in zarr_array[group]:
            return None

        label = torch.from_numpy(zarr_array[group][version][y:y+self.tile_size, x:x+self.tile_size]).float()

        return label

    def _get_mask(self, zarr_array, x, y):
        if "mask" not in zarr_array or self.use_as_unlabeled:
            return None

        return torch.from_numpy(zarr_array["mask"][y:y+self.tile_size, x:x+self.tile_size]).float()

    def _get_pseudo_mask(self, zarr_array, x, y):
        if self.pseudo_label_version is None:
            return None

        group = "pseudo_masks"
        if group not in zarr_array:
            return None

        version = str(self.pseudo_label_version)
        if version not in zarr_array[group]:
            return None

        return torch.from_numpy(zarr_array[group][version][y:y+self.tile_size, x:x+self.tile_size]).float()

    def _get_tiles_for_segment(self, segment, min_mask_fraction):
        segment_path = self.segment_paths[segment]

        if self.pseudo_label_version is not None:
            tile_dir = segment_path / "pseudo_tiles" / str(self.pseudo_label_version) / self.pseudo_target
            tile_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else "None"}.npy"
            csv_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else "None"}.csv"
        else:
            tile_path = segment_path / "tiles" / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else "None"}.npy"
            csv_path = segment_path / "tiles" / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else "None"}.csv"
        
        # if not tile_path.exists():
        #     raise FileNotFoundError(f"Missing tile index: {tile_path}")

        if tile_path.exists():
            tiles = np.load(tile_path, allow_pickle=False)
        else:
            print(f"Missing tile index: {tile_path}. Creating tiles.")

            zarr_data = self.zarr_data[segment]
            volume_shape = zarr_data["volume"].shape
            h, w = volume_shape[-2:]

            label = np.asarray(zarr_data["label"]) if "label" in zarr_data else None
            mask = np.asarray(zarr_data["mask"]) if "mask" in zarr_data else None

            if self.pseudo_label_version is not None:
                version = str(self.pseudo_label_version)
                group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
                label = np.asarray(zarr_data[group][version])
                pseudo_mask = np.asarray(zarr_data["pseudo_masks"][version])

                if mask is not None:
                    mask = mask * pseudo_mask
                else:
                    mask = pseudo_mask

            validation_mask = np.asarray(zarr_data["validation_mask"]) if "validation_mask" in zarr_data else None
            if self.split is not None and validation_mask is not None:
                if self.split == "train":
                    validation_mask = validation_mask == 0
                mask = mask * validation_mask

            tiles = get_tiles(label, mask, self.tile_size, self.stride, segment, h, w)

            tile_path.parent.mkdir(parents=True, exist_ok=True)
            save_tiles(tiles, tile_path, csv_path)

        tile_list = []

        for t in tiles:
            if self.use_as_unlabeled and self.pseudo_label_version is None:
                positive_fraction = float("nan")
                mask_fraction = float("nan")
            else:
                positive_fraction = float(t["positive_fraction"])
                mask_fraction = float(t["mask_fraction"])

            if not np.isnan(mask_fraction) and mask_fraction < min_mask_fraction:
                continue
            
            tile_list.append(
                {
                    "segment": t["segment"],
                    "y": int(t["y"]),
                    "x": int(t["x"]),
                    "positive_fraction": positive_fraction,
                    "mask_fraction": mask_fraction,
                }
            )

        return tile_list

    def _get_zarr_data_for_segment(self, segment_path):
        zarr_path = segment_path / f"{segment_path.name}.zarr"
        
        if not zarr_path.exists():
            raise FileNotFoundError(f"Missing zarr archive: {zarr_path}")

        zarr_data = zarr.open(zarr_path, mode="r")

        # if self.split is not None and "validation_mask" not in zarr_data:
        #     raise ValueError("Split can only be used if the segment contains a validation mask")

        if self.pseudo_label_version is not None:
            group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
            if group not in zarr_data:
                raise KeyError(f"Missing {self.pseudo_target} pseudo targets ('{group}') in {zarr_path}.")

            # if self.pseudo_label_version is None:
            #     raise ValueError("pseudo_label_version must be specified when use_pseudo_labels=True")

            # pseudo_targets = zarr_data[group]
            # available_versions = list(pseudo_targets.keys())

            # version = str(self.pseudo_label_version)
            # if version not in pseudo_targets:
            #     raise KeyError(f"Missing pseudo target version '{version}'. Available versions: {available_versions}")

            version = str(self.pseudo_label_version)

            available_versions = list(zarr_data[group].keys())
            if version not in zarr_data[group]:
                raise KeyError(f"Missing pseudo target version '{version}'. Available versions: {available_versions}")

            if "pseudo_masks" not in zarr_data:
                raise KeyError(f"Missing 'pseudo_masks' in {zarr_path}.")
            
            available_mask_versions = list(zarr_data["pseudo_masks"].keys())
            if version not in zarr_data["pseudo_masks"]:
                raise KeyError(f"Missing pseudo mask version '{version}'. Available versions: {available_mask_versions}")

        return zarr_data

    def _get_meta_for_segment(self, segment_path):
        meta_path = segment_path / "metadata.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return meta


    def _assert_binary(self, tensor, name):
        values = torch.unique(tensor)
        assert torch.all((values == 0) | (values == 1)), (f"{name} is not binary. Found values: {values[:20].tolist()}")
