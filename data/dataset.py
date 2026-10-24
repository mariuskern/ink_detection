from pathlib import Path
import json

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, ConcatDataset
from torchvision import tv_tensors

from utils.tiles import get_tiles, save_tiles
from utils.segment_paths import get_segment_paths
from utils.create_training_mask import create_training_mask


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
    
    segment_crops:
        Optional dictionary specifying a spatial crop for individual segments.
        The crop is defined as `(y_start, y_end, x_start, x_end)` in the original
        segment coordinates. Only tiles that lie completely within the
        specified crop are used. Segments without a specified crop are used
        without spatial restriction.

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

    mask_refinement:
        Optional configuration for label-neighborhood supervision. targets
        selects real and/or pseudo labels (default: real). None disables it.
        Unlabeled data without pseudo targets is never refined. Saved masks
        intersect only the corresponding target validity; the split is applied on reading. They filter
        tiles before augmentation.
    """

    def __init__(
        self,
        data_dir,
        segments=None,
        segment_crops=None,
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
        split=None,
        mask_refinement=None,
        intersect_pseudo_with_original_mask=False,
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
        self.segment_crops = segment_crops or {}
        self.mask_refinement = mask_refinement or {}
        self.intersect_pseudo_with_original_mask = intersect_pseudo_with_original_mask

        self.z0 = self.center - self.depth // 2
        self.z1 = self.z0 + self.depth

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

        self.refinement_params = {"margin": self.mask_refinement["margin"]} if self.mask_refinement.get("margin") is not None else {"margin_factor": self.mask_refinement.get("margin_factor", 0.2)}
        if self.mask_refinement.get("closing_radius") is not None:
            self.refinement_params["closing_radius"] = self.mask_refinement["closing_radius"]
        else:
            self.refinement_params["closing_factor"] = self.mask_refinement.get("closing_factor") or 0
        self.refinement_params.update(method=self.mask_refinement.get("closing_method", "auto"), chunk=self.mask_refinement.get("chunk_size", 2048))
        if self.pseudo_label_version is not None and self.pseudo_target == "soft":
            self.refinement_params["threshold"] = self.mask_refinement.get("pseudo_threshold", 0.5)

        refinement_targets = (mask_refinement or {}).get("targets", ["real"])
        if isinstance(refinement_targets, str) or any(t not in ("real", "pseudo") for t in refinement_targets):
            raise ValueError("mask_refinement.targets must be a list containing real and/or pseudo")

        for segment in self.segments:
            if segment not in self.segment_paths:
                raise KeyError(f"Unknown segment '{segment}'. Available segments: {sorted(self.segment_paths.keys())}")

            segment_path = self.segment_paths[segment]

            # Load zarr data
            self.zarr_data[segment] = self._get_zarr_data_for_segment(segment_path)

            volume_depth = self.zarr_data[segment]["volume"].shape[0]
            if not 0 <= self.z0 < self.z1 <= volume_depth:
                raise ValueError(f"Invalid depth range [{self.z0}:{self.z1}] for segment '{segment}' with {volume_depth} layers (center={self.center}, depth={self.depth}).")

            refine_real = not self.use_as_unlabeled and "label" in self.zarr_data[segment] and "real" in refinement_targets
            refine_pseudo = self.pseudo_label_version is not None and "pseudo" in refinement_targets

            if mask_refinement and mask_refinement.get("enabled", False):
                if refine_real:
                    self._prepare_refined_mask(segment_path, self.zarr_data[segment], mask_refinement)

                if refine_pseudo:
                    self._prepare_refined_mask(segment_path, self.zarr_data[segment], mask_refinement, pseudo=True)

                if refine_real or refine_pseudo:
                    self.zarr_data[segment] = self._get_zarr_data_for_segment(segment_path)

            # Load tiles
            self.tiles.extend(self._get_tiles_for_segment(segment, min_mask_fraction, crop=self.segment_crops.get(segment)))

            # Load meta
            self.segment_shapes[segment] = torch.tensor(self._get_meta_for_segment(segment_path)["volume_shape"])

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

        # tile = tv_tensors.Image(tile)
        # label = tv_tensors.Mask(label)
        # pseudo_label = tv_tensors.Mask(pseudo_label)
        # mask = tv_tensors.Mask(mask)
        # pseudo_mask = tv_tensors.Mask(pseudo_mask)

        if self.transform is not None:
            tile, label, pseudo_label, mask, pseudo_mask = self.transform(tile, label, pseudo_label, mask, pseudo_mask)

        tile = tile.unsqueeze(0) # [1, D, H, W]
        label = label.unsqueeze(0) # [1, H, W]
        pseudo_label = pseudo_label.unsqueeze(0)
        mask = mask.unsqueeze(0)
        pseudo_mask = pseudo_mask.unsqueeze(0)

        # if self.transform is not None:
        #     tile, label, pseudo_label, mask, pseudo_mask = self.transform(tile, label, pseudo_label, mask, pseudo_mask)

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

    def _apply_validation_mask(self, mask, zarr_array, x, y):
        if self.split is not None and "validation_mask" in zarr_array:
            validation = zarr_array["validation_mask"][y:y+self.tile_size, x:x+self.tile_size] > 0
            valid = ~validation if self.split == "train" else validation

            mask = valid if mask is None else mask * valid
        
        return torch.from_numpy(np.asarray(mask)).float() if mask is not None else None

    def _get_mask(self, zarr_array, x, y):
        if self.use_as_unlabeled:
            return None
        
        source = self._get_refined_mask(zarr_array)

        if source is None:
            source = zarr_array.get("mask")

        mask = source[y:y+self.tile_size, x:x+self.tile_size] if source is not None else None

        return self._apply_validation_mask(mask, zarr_array, x, y)

    def _get_pseudo_mask(self, zarr_array, x, y):
        if self.pseudo_label_version is None:
            return None
        
        version = str(self.pseudo_label_version)
        source = self._get_refined_mask(zarr_array, pseudo=True)

        if source is None:
            source = zarr_array.get("pseudo_masks", {}).get(version)

        if source is None:
            return None
        
        mask = source[y:y+self.tile_size, x:x+self.tile_size]

        if self.intersect_pseudo_with_original_mask and "mask" in zarr_array:
            mask = mask * zarr_array["mask"][y:y+self.tile_size, x:x+self.tile_size]

        return self._apply_validation_mask(mask, zarr_array, x, y)

    def _get_tiles_for_segment(self, segment, min_mask_fraction, crop=None):
        segment_path = self.segment_paths[segment]
        zarr_data = self.zarr_data[segment]

        refined_real_mask = self._get_refined_mask(zarr_data, pseudo=False)
        refined_pseudo_mask = self._get_refined_mask(zarr_data, pseudo=True)

        use_pseudo = self.pseudo_label_version is not None
        has_refined_mask = refined_pseudo_mask is not None if use_pseudo else refined_real_mask is not None

        if has_refined_mask:
            refinement_suffix = "_".join(f"{key}_{value}" for key, value in sorted(self.refinement_params.items()))
            tile_folder = f"refined_tiles_{refinement_suffix}"
            pseudo_tile_folder = f"refined_pseudo_tiles_{refinement_suffix}"
        else:
            tile_folder = "tiles"
            pseudo_tile_folder = "pseudo_tiles"

        if use_pseudo:
            tile_dir = segment_path / pseudo_tile_folder / str(self.pseudo_label_version) / self.pseudo_target
            tile_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else 'None'}.npy"
            csv_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else 'None'}.csv"
        else:
            tile_dir = segment_path / tile_folder
            tile_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else 'None'}.npy"
            csv_path = tile_dir / f"tile_size_{self.tile_size}_stride_{self.stride}_split_{self.split if self.split is not None else 'None'}.csv"

        tile_path = tile_path.with_stem(f"{tile_path.stem}_use_as_unlabeled_{self.use_as_unlabeled}")
        csv_path = csv_path.with_stem(f"{csv_path.stem}_use_as_unlabeled_{self.use_as_unlabeled}")

        if use_pseudo:
            tile_path = tile_path.with_stem(f"{tile_path.stem}_intersect_original_{self.intersect_pseudo_with_original_mask}")
            csv_path = csv_path.with_stem(f"{csv_path.stem}_intersect_original_{self.intersect_pseudo_with_original_mask}")

        # if not tile_path.exists():
        #     raise FileNotFoundError(f"Missing tile index: {tile_path}")

        if tile_path.exists():
            tiles = np.load(tile_path, allow_pickle=False)
        else:
            print(f"Missing tile index: {tile_path}. Creating tiles.")

            volume_shape = zarr_data["volume"].shape
            h, w = volume_shape[-2:]

            original_mask = np.asarray(zarr_data["mask"]) if "mask" in zarr_data else None

            if use_pseudo:
                version = str(self.pseudo_label_version)
                group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
                
                label = np.asarray(zarr_data[group][version])

                pseudo_mask = np.asarray(refined_pseudo_mask if refined_pseudo_mask is not None else zarr_data["pseudo_masks"][version])

                if self.intersect_pseudo_with_original_mask and original_mask is not None:
                    mask = original_mask * pseudo_mask
                else:
                    mask = pseudo_mask
            else:
                label = np.asarray(zarr_data["label"]) if "label" in zarr_data and not self.use_as_unlabeled else None

                if self.use_as_unlabeled:
                    mask = None
                else:
                    mask = np.asarray(refined_real_mask) if refined_real_mask is not None else original_mask

            validation_mask = np.asarray(zarr_data["validation_mask"]) if "validation_mask" in zarr_data else None

            if self.split is not None and validation_mask is not None:
                if self.split == "train":
                    validation_mask = validation_mask == 0
                mask = mask * validation_mask if mask is not None else validation_mask

            tiles = get_tiles(label, mask, self.tile_size, self.stride, segment, h, w)

            tile_path.parent.mkdir(parents=True, exist_ok=True)
            save_tiles(tiles, tile_path, csv_path)

        if crop is not None:
            # x_min, y_min, x_max, y_max = crop
            y_start, y_end, x_start, x_end = crop

            tiles = [t for t in tiles if (
                x_start <= int(t["x"])
                and int(t["x"]) + self.tile_size <= x_end
                and y_start <= int(t["y"])
                and int(t["y"]) + self.tile_size <= y_end
            )]

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

    def _refined_mask_key(self, settings, pseudo=False):
        """Use the same parameter-derived path for writing and reading."""

        version = str(self.pseudo_label_version) if pseudo else None
        group_path = "refined_masks/pseudo" if pseudo else "refined_masks/real"

        if pseudo:
            group_path += f"/{version}/{self.pseudo_target}"

        margin = settings.get("margin")
        margin_name = f"margin_{margin}" if margin is not None else f"margin_factor_{settings.get('margin_factor', 0.2)}"

        closing = settings.get("closing_radius")
        closing_name = f"closing_radius_{closing}" if closing is not None else f"closing_factor_{settings.get('closing_factor') or 0}"

        name = f"{margin_name}_{closing_name}_method_{settings.get('closing_method', 'auto')}_chunk_{settings.get('chunk_size', 2048)}"

        if pseudo and self.pseudo_target == "soft":
            name += f"_threshold_{settings.get('pseudo_threshold', 0.5)}"

        mask_key = f"{group_path}/{name}"

        return mask_key

    def _get_refined_mask(self, archive, pseudo=False):
        settings = self.mask_refinement

        target = "pseudo" if pseudo else "real"

        if not settings.get("enabled", False) or target not in settings.get("targets", ["real"]):
            return None
        
        if pseudo:
            if self.pseudo_label_version is None:
                return None
        elif self.use_as_unlabeled or "label" not in archive:
            return None
        
        key = self._refined_mask_key(settings, pseudo=pseudo)
        mask = archive.get(key)

        return mask

    def _prepare_refined_mask(self, segment_path, archive, settings, pseudo=False):
        """Create the final validity mask inside the segment's Zarr archive."""

        version = str(self.pseudo_label_version) if pseudo else None
        target = f"{'pseudo_labels' if self.pseudo_target == 'hard' else 'prediction'}/{version}" if pseudo else "label"
        labels = np.asarray(archive[target])

        if pseudo and self.pseudo_target == "soft":
            threshold = settings.get("pseudo_threshold", 0.5)

            if not np.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("pseudo_threshold must be in (0, 1]")

            labels = (labels >= threshold).astype(np.uint8)

        mask_key = self._refined_mask_key(settings, pseudo=pseudo)
        group_path, name = mask_key.rsplit("/", 1)
        zarr_path = segment_path / f"{segment_path.name}.zarr"

        if mask_key in archive:
            return

        chunk_size = settings.get("chunk_size", 2048)

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        print(f"Creating refined {'pseudo' if pseudo else 'real'} mask: {zarr_path}/{mask_key}")

        writable_archive = zarr.open_group(str(zarr_path), mode="a")
        group = writable_archive.require_group(group_path)
        create_array = group.create_array if hasattr(group, "create_array") else group.create_dataset
        mask = create_array(name, shape=labels.shape, chunks=(min(chunk_size, 256),) * 2, dtype="uint8", overwrite=True)

        options = {key: settings[key] for key in ("margin", "margin_factor", "closing_radius", "closing_factor", "closing_method", "chunk_size", "progress") if key in settings}

        valid = create_training_mask(labels, **options) > 0

        if not pseudo and "mask" in archive:
            valid &= np.asarray(archive["mask"]) > 0

        if pseudo:
            valid &= np.asarray(archive[f"pseudo_masks/{version}"]) > 0

        mask[:] = valid.astype(np.uint8)

        return
