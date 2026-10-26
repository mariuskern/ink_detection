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
from utils.refine_labels import refine_labels


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
    - a tile index used to determine valid tiles

    Versioned pseudo targets, masks and their refinements are stored separately
    in the run's temporary directory, supplied via pseudo_cache_dir.

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

    pseudo_positive_threshold:
        Pixel probability threshold for positive tile fractions with soft
        pseudo targets. Only affects tile statistics, not training targets.

    transform:
        Optional transformation applied jointly to the image, labels and
        masks.

    use_original_mask_for_pseudo_mask:
        Intersect pseudo_mask with the original mask and restrict pseudo
        tile selection accordingly. With use_as_unlabeled=True and no pseudo
        label version, only restricts tile selection; no masks are loaded.

    mask_refinement:
        Optional configuration for label-neighborhood supervision. targets
        selects real and/or pseudo labels (default: real). None disables it.
        Unlabeled data without pseudo targets is never refined. Saved masks
        intersect only the corresponding target validity; the split is applied on reading. They filter
        tiles before augmentation.

    label_refinement:
        Optional configuration for soft targets; targets selects real and/or
        pseudo labels (default: real). Soft pseudo targets are not supported.
        Results are cached in Zarr;
        tile selection and validity masks use original labels.
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
        use_original_mask_for_pseudo_mask=False,
        label_refinement=None,
        pseudo_cache_dir=None,
        pseudo_positive_threshold=0.5,
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
        self.pseudo_positive_threshold = pseudo_positive_threshold
        self.transform = transform
        self.split = split
        self.segment_crops = segment_crops or {}
        self.mask_refinement = mask_refinement or {}
        self.use_original_mask_for_pseudo_mask = use_original_mask_for_pseudo_mask
        self.label_refinement = label_refinement or {}
        self.pseudo_cache_dir = Path(pseudo_cache_dir) if pseudo_cache_dir is not None else None
        
        if self.pseudo_label_version is not None:
            if self.pseudo_cache_dir is None:
                raise ValueError("pseudo_cache_dir is required when pseudo_label_version is set and must match the directory used to save pseudo Zarr archives.")
            
            self.pseudo_cache_dir.mkdir(parents=True, exist_ok=True)

        self.z0 = self.center - self.depth // 2
        self.z1 = self.z0 + self.depth

        if self.norm not in ["global", "local"]:
            raise ValueError(f"Invalid normalization mode: {self.norm}. Must be 'global' or 'local'.")

        if self.pseudo_target not in ["hard", "soft"]:
            raise ValueError(f"Invalid pseudo target '{self.pseudo_target}'. Must be 'hard' or 'soft'.")

        if split is not None and split not in ["train", "val"]:
            raise ValueError(f"Invalid split '{self.split}'. Must be None, 'train' or 'val'.")

        # if use_as_unlabeled and pseudo_label_version is None:
        #     raise ValueError("use_as_unlabeled=True requires pseudo_label_version.")

        self.segment_paths = get_segment_paths(self.data_dir)

        if segments is None:
            self.segments = sorted(self.segment_paths.keys())
        else:
            self.segments = list(map(str, segments))

        self.tiles = []
        self.segment_archives = {}
        self.segment_shapes = {}
        self.pseudo_archives = {}

        refinement_targets = (self.mask_refinement or {}).get("targets", ["real"])
        if isinstance(refinement_targets, str) or any(t not in ("real", "pseudo") for t in refinement_targets):
            raise ValueError("mask_refinement.targets must be a list containing real and/or pseudo")

        label_targets = self.label_refinement.get("targets", ["real"])
        if isinstance(label_targets, str) or any(t not in ("real", "pseudo") for t in label_targets):
            raise ValueError("label_refinement.targets must be a list containing real and/or pseudo")
        
        if (self.label_refinement.get("enabled", False) and "pseudo" in label_targets and self.pseudo_label_version is not None and self.pseudo_target == "soft"):
            raise ValueError("Label refinement does not support soft pseudo labels. Use pseudo_target='hard' or remove 'pseudo' from label_refinement.targets.")

        for segment in self.segments:
            if segment not in self.segment_paths:
                raise KeyError(f"Unknown segment '{segment}'. Available segments: {sorted(self.segment_paths.keys())}")

            segment_path = self.segment_paths[segment]

            # Load zarr data
            self.segment_archives[segment] = self._open_segment_archive(segment)
            segment_archive = self.segment_archives[segment]
            
            if self.pseudo_label_version is not None:
                self.pseudo_archives[segment] = self._open_pseudo_archive(segment)
                pseudo_archive = self.pseudo_archives[segment]
                group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"

                for key in (f"{group}/{self.pseudo_label_version}", f"pseudo_masks/{self.pseudo_label_version}"):
                    if key not in pseudo_archive:
                        raise KeyError(f"Missing pseudo target '{key}' in {self.pseudo_cache_dir / segment}.")

            volume_depth = segment_archive["volume"].shape[0]
            if not 0 <= self.z0 < self.z1 <= volume_depth:
                raise ValueError(f"Invalid depth range [{self.z0}:{self.z1}] for segment '{segment}' with {volume_depth} layers (center={self.center}, depth={self.depth}).")

            refine_real = not self.use_as_unlabeled and "label" in segment_archive and "real" in refinement_targets
            refine_pseudo = self.pseudo_label_version is not None and "pseudo" in refinement_targets

            if self.mask_refinement and self.mask_refinement.get("enabled", False):
                if refine_real:
                    self._prepare_refined_mask(segment_archive)

                if refine_pseudo:
                    self._prepare_refined_mask(pseudo_archive, pseudo=True)

            # Load tiles
            self.tiles.extend(self._get_tiles_for_segment(segment, min_mask_fraction, crop=self.segment_crops.get(segment)))

            if self.label_refinement.get("enabled", False):
                if "real" in label_targets and not self.use_as_unlabeled and "label" in segment_archive:
                    self._prepare_refined_label(segment_archive)

                if "pseudo" in label_targets and self.pseudo_label_version is not None:
                    self._prepare_refined_label(pseudo_archive, pseudo=True)

            # Load meta
            self.segment_shapes[segment] = torch.tensor(self._get_meta_for_segment(segment_path)["volume_shape"])

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        entry = self.tiles[idx]

        segment = entry["segment"]
        x = entry["x"]
        y = entry["y"]

        tile = self._get_tile(segment, x, y)
        label = self._get_label(segment, x, y)
        pseudo_label = self._get_pseudo_label(segment, x, y)
        mask = self._get_mask(segment, x, y)
        pseudo_mask = self._get_pseudo_mask(segment, x, y)

        if label is not None and not self._refines_labels():
            self._assert_binary(label, "label")

        if pseudo_label is not None and self.pseudo_target == "hard" and not self._refines_labels(pseudo=True):
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

    def _get_tile(self, segment, x, y):
        zarr_array = self.segment_archives[segment]

        tile = torch.from_numpy(zarr_array["volume"][self.z0:self.z1, y:y+self.tile_size, x:x+self.tile_size]).float()
        tile = self._normalize(tile, mode=self.norm)

        return tile

    def _get_label(self, segment, x, y):
        zarr_array = self.segment_archives[segment]

        if "label" not in zarr_array or self.use_as_unlabeled:
            return None

        key = self._refined_label_key() if self._refines_labels() else "label"
        label = torch.from_numpy(zarr_array[key][y:y+self.tile_size, x:x+self.tile_size]).float()

        return label

    def _get_pseudo_label(self, segment, x, y):
        if self.pseudo_label_version is None:
            return None

        zarr_array = self.pseudo_archives[segment]
        group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
        
        if group not in zarr_array:
            return None

        version = str(self.pseudo_label_version)
        if version not in zarr_array[group]:
            return None

        source = zarr_array[self._refined_label_key(pseudo=True)] if self._refines_labels(pseudo=True) else zarr_array[group][version]
        label = torch.from_numpy(source[y:y+self.tile_size, x:x+self.tile_size]).float()

        return label

    def _apply_validation_mask(self, mask, zarr_array, x, y):
        if self.split is not None and "validation_mask" in zarr_array:
            validation = zarr_array["validation_mask"][y:y+self.tile_size, x:x+self.tile_size] > 0
            valid = ~validation if self.split == "train" else validation

            mask = valid if mask is None else mask * valid
        
        return torch.from_numpy(np.asarray(mask)).float() if mask is not None else None

    def _get_mask(self, segment, x, y):
        if self.use_as_unlabeled:
            return None
        
        zarr_array = self.segment_archives[segment]
        source = self._get_refined_mask(segment)

        if source is None:
            source = zarr_array.get("mask")

        mask = source[y:y+self.tile_size, x:x+self.tile_size] if source is not None else None

        return self._apply_validation_mask(mask, zarr_array, x, y)

    def _get_pseudo_mask(self, segment, x, y):
        if self.pseudo_label_version is None:
            return None
        
        zarr_array = self.segment_archives[segment]

        version = str(self.pseudo_label_version)
        source = self._get_refined_mask(segment, pseudo=True)

        if source is None:
            source = self.pseudo_archives[segment].get("pseudo_masks", {}).get(version)

        if source is None:
            return None
        
        mask = source[y:y+self.tile_size, x:x+self.tile_size]

        if self.use_original_mask_for_pseudo_mask and "mask" in zarr_array:
            mask = mask * zarr_array["mask"][y:y+self.tile_size, x:x+self.tile_size]

        return self._apply_validation_mask(mask, zarr_array, x, y)

    def _get_tiles_for_segment(self, segment, min_mask_fraction, crop=None):
        segment_path = self.segment_paths[segment]
        zarr_data = self.segment_archives[segment]

        refined_real_mask = self._get_refined_mask(segment, pseudo=False)
        refined_pseudo_mask = self._get_refined_mask(segment, pseudo=True)

        use_pseudo = self.pseudo_label_version is not None

        if not use_pseudo:
            tile_folder = "tiles"
            if refined_real_mask is not None:
                suffix = self._refined_mask_key().rsplit("/", 1)[1]
                tile_folder = f"refined_tiles_{suffix}"

            tile_dir = segment_path / tile_folder
            
            stem = (
                f"tile_size_{self.tile_size}_stride_{self.stride}"
                f"_split_{self.split}_use_as_unlabeled_{self.use_as_unlabeled}"
            )
            
            if self.use_as_unlabeled:
                stem += f"_original_mask_{self.use_original_mask_for_pseudo_mask}"

            tile_path = tile_dir / f"{stem}.npy"
            csv_path = tile_dir / f"{stem}.csv"

        if not use_pseudo and tile_path.exists():
            tiles = np.load(tile_path, allow_pickle=False)
        else:
            print(f"Creating pseudo tiles for {segment}." if use_pseudo else f"Missing tile index: {tile_path}. Creating tiles.")

            volume_shape = zarr_data["volume"].shape
            h, w = volume_shape[-2:]

            original_mask = np.asarray(zarr_data["mask"]) if "mask" in zarr_data else None

            if use_pseudo:
                version = str(self.pseudo_label_version)
                group = "pseudo_labels" if self.pseudo_target == "hard" else "prediction"
                
                pseudo_archive = self.pseudo_archives[segment]
                label = np.asarray(pseudo_archive[group][version])

                pseudo_mask = np.asarray(refined_pseudo_mask if refined_pseudo_mask is not None else pseudo_archive["pseudo_masks"][version])

                if self.use_original_mask_for_pseudo_mask and original_mask is not None:
                    mask = original_mask * pseudo_mask
                else:
                    mask = pseudo_mask
            elif self.use_as_unlabeled:
                label = None
                mask = original_mask if self.use_original_mask_for_pseudo_mask else None
            else:
                label = np.asarray(zarr_data["label"]) if "label" in zarr_data else None
                mask = np.asarray(refined_real_mask) if refined_real_mask is not None else original_mask

            validation_mask = np.asarray(zarr_data["validation_mask"]) if "validation_mask" in zarr_data else None

            if self.split is not None and validation_mask is not None:
                if self.split == "train":
                    validation_mask = validation_mask == 0
                mask = mask * validation_mask if mask is not None else validation_mask

            tiles = get_tiles(
                label, mask, self.tile_size, self.stride, segment, h, w,
                positive_threshold=self.pseudo_positive_threshold if use_pseudo and self.pseudo_target == "soft" else None,
            )

            if not use_pseudo:
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
            mask_fraction = float(t["mask_fraction"])

            positive_fraction = float(t["positive_fraction"])

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

    def _open_segment_archive(self, segment):
        segment_path = self.segment_paths[segment]
        path = segment_path / f"{segment_path.name}.zarr"

        if not path.exists():
            raise FileNotFoundError(f"Missing zarr archive: {path}")
        
        return zarr.open_group(str(path), mode="a")

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

    def _label_refinement_options(self):
        settings = self.label_refinement

        return {
            "radius": settings.get("radius"),
            "radius_factor": settings.get("radius_factor", 0.2),
            "min_value": settings.get("min_value", 0.5),
            "smooth_background": settings.get("smooth_background", False),
            "chunk_size": settings.get("chunk_size", 2048),
            "progress": settings.get("progress", False),
        }

    def _refines_labels(self, pseudo=False):
        return self.label_refinement.get("enabled", False) and ("pseudo" if pseudo else "real") in self.label_refinement.get("targets", ["real"])

    def _refined_label_key(self, pseudo=False):
        options = self._label_refinement_options()

        # Progress only affects display, not the stored target values.
        name = "_".join(f"{key}_{value}" for key, value in sorted(options.items()) if key != "progress")

        if pseudo:
            return f"refined_labels/pseudo/{self.pseudo_label_version}/{self.pseudo_target}/{name}"
        
        return f"refined_labels/real/{name}"

    def _open_pseudo_archive(self, segment):
        path = self.pseudo_cache_dir / segment / f"{segment}.zarr"

        if not path.exists():
            raise FileNotFoundError(f"Missing pseudo Zarr archive: {path}")
        
        return zarr.open_group(str(path), mode="a")

    def _prepare_refined_label(self, archive, pseudo=False):
        key = self._refined_label_key(pseudo=pseudo)

        if key in archive:
            return

        options = self._label_refinement_options()
        chunk_size = options["chunk_size"]

        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("label_refinement.chunk_size must be a positive integer")

        source_key = f"{'pseudo_labels' if self.pseudo_target == 'hard' else 'prediction'}/{self.pseudo_label_version}" if pseudo else "label"
        source = archive[source_key]

        print(f"Creating refined label: {key}")

        group_path, name = key.rsplit("/", 1)
        group = archive.require_group(group_path)
        create_array = group.create_array if hasattr(group, "create_array") else group.create_dataset

        target = create_array(name, shape=source.shape, chunks=(min(chunk_size, 256),) * 2, dtype="float32")
        
        try:
            target[:] = refine_labels(source, **options)
        except BaseException:
            # An interrupted/failed calculation must not become a valid cache.
            del group[name]
            raise

    def _mask_refinement_options(self):
        """Effective create_training_mask arguments, shared by calculation and caches."""

        settings = self.mask_refinement

        return {
            "margin": settings.get("margin"),
            "margin_factor": settings.get("margin_factor", 0.2),
            "closing": settings.get("closing"),
            "closing_factor": settings.get("closing_factor") or 0,
            "closing_method": settings.get("closing_method", "auto"),
            "chunk_size": settings.get("chunk_size", 2048),
            "progress": settings.get("progress", True),
        }

    def _refines_masks(self, pseudo=False):
        return self.mask_refinement.get("enabled", False) and ("pseudo" if pseudo else "real") in self.mask_refinement.get("targets", ["real"])

    def _refined_mask_key(self, pseudo=False):
        options = self._mask_refinement_options()

        if pseudo and self.pseudo_target == "soft":
            options["pseudo_threshold"] = self.mask_refinement.get("pseudo_threshold", 0.5)

        name = "_".join(f"{key}_{value}" for key, value in sorted(options.items()) if key != "progress")

        if pseudo:
            return f"refined_masks/pseudo/{self.pseudo_label_version}/{self.pseudo_target}/{name}"

        return f"refined_masks/real/{name}"

    def _get_refined_mask(self, segment, pseudo=False):
        archive = self.segment_archives[segment]

        if not self._refines_masks(pseudo=pseudo):
            return None
        
        if pseudo:
            if self.pseudo_label_version is None:
                return None
        elif self.use_as_unlabeled or "label" not in archive:
            return None
        
        key = self._refined_mask_key(pseudo=pseudo)
        source = self.pseudo_archives.get(segment) if pseudo else archive
        mask = source.get(key) if source is not None else None

        return mask

    def _prepare_refined_mask(self, archive, pseudo=False):
        """Store the refined validity mask in the source archive."""

        mask_key = self._refined_mask_key(pseudo=pseudo)

        if mask_key in archive:
            return

        version = str(self.pseudo_label_version) if pseudo else None
        target = f"{'pseudo_labels' if self.pseudo_target == 'hard' else 'prediction'}/{version}" if pseudo else "label"
        labels = np.asarray(archive[target])

        if pseudo and self.pseudo_target == "soft":
            threshold = self.mask_refinement.get("pseudo_threshold", 0.5)

            if not np.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("pseudo_threshold must be in (0, 1]")

            labels = (labels >= threshold).astype(np.uint8)

        group_path, name = mask_key.rsplit("/", 1)

        options = self._mask_refinement_options()
        chunk_size = options["chunk_size"]

        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        print(f"Creating refined {'pseudo' if pseudo else 'real'} mask: {mask_key}")

        group = archive.require_group(group_path)
        create_array = group.create_array if hasattr(group, "create_array") else group.create_dataset
        mask = create_array(name, shape=labels.shape, chunks=(min(chunk_size, 256),) * 2, dtype="uint8")

        try:
            valid = create_training_mask(labels, **options) > 0

            if not pseudo and "mask" in archive:
                valid &= np.asarray(archive["mask"]) > 0

            if pseudo:
                valid &= np.asarray(archive[f"pseudo_masks/{version}"]) > 0

            mask[:] = valid.astype(np.uint8)
        except BaseException:
            # Failed calculations, writes and user interrupts must not become a valid cache.
            del group[name]
            raise

        return
