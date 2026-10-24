import math
import operator

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


def _binary(block):
    block = np.asarray(block)

    if block.ndim == 3:
        block = cv2.cvtColor(block, cv2.COLOR_RGB2GRAY)
    return (block > 0).astype(np.uint8)


def _blocks(height, width, chunk_size, *, progress, desc):
    total = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)
    coordinates = ((y, x) for y in range(0, height, chunk_size) for x in range(0, width, chunk_size))

    with tqdm(coordinates, total=total, desc=desc, unit="block", disable=not progress) as blocks:
        yield from blocks


def estimate_component_size(label, chunk_size=2048, *, progress=True):
    """Estimate typical max(width, height) using the median of components.

    Scan blocks with 8-connectivity. Components cut by an internal block
    boundary are excluded rather than counted as smaller fragments. Thus
    this is an estimate, not a global connected-component analysis. Choose
    blocks much larger than letters. All blocks are scanned, including for
    sparse segments. A size histogram keeps memory independent of the total
    component count. Raises if foreground exists but no complete component
    can be measured; returns 0 for an empty label.
    """

    chunk_size = operator.index(chunk_size)

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    height, width = label.shape[:2]
    histogram = np.zeros(chunk_size + 1, dtype=np.int64)
    has_foreground = False

    for y, x in _blocks(height, width, chunk_size, progress=progress, desc="Estimating component size"):
        bottom = min(height, y + chunk_size)
        right = min(width, x + chunk_size)
        block = _binary(label[y:bottom, x:right])

        if not block.any():
            continue

        has_foreground = True
        _, components, stats, _ = cv2.connectedComponentsWithStats(block, connectivity=8)
        keep = np.ones(len(stats), dtype=bool)
        keep[0] = False

        # Actual image borders do not truncate components; internal ones do.
        if y > 0:
            keep[np.unique(components[0, :])] = False

        if bottom < height:
            keep[np.unique(components[-1, :])] = False

        if x > 0:
            keep[np.unique(components[:, 0])] = False

        if right < width:
            keep[np.unique(components[:, -1])] = False
        sizes = stats[keep, 2:4].max(axis=1)
        histogram += np.bincount(sizes, minlength=len(histogram))
    total = int(histogram.sum())

    if not total:
        if has_foreground:
            raise ValueError("No complete components found; increase chunk_size or set margin explicitly")

        return 0.0

    cumulative = histogram.cumsum()
    lower = np.searchsorted(cumulative, (total - 1) // 2 + 1)
    upper = np.searchsorted(cumulative, total // 2 + 1)
    
    return float((lower + upper) / 2)


def create_training_mask(
    label,
    margin_factor=0.2,
    *,
    margin=None,
    chunk_size=2048,
    closing=None,
    closing_factor=None,
    closing_method="auto",
    progress=True,
):
    """Return a uint8 mask (0/255) around positive labels.

    Set progress=False to disable per-block progress bars for both stages.

    Without an explicit margin, estimate the typical component size with
    estimate_component_size and use int(size * margin_factor) as a uniform
    Euclidean pixel radius. The median resists unusually small/large objects
    as long as they do not dominate the component count. A zero factor gives
    zero radius; positive factors give at least one pixel for nonempty input.

    Both modes use overlapping blocks. ``label`` may be a sliceable array
    such as Zarr or numpy.memmap. The returned mask has shape label.shape[:2]
    and is allocated in RAM. Set ``closing`` to a positive integer to close small gaps
    in the expanded training mask with an elliptical kernel of diameter
    2 * closing + 1. Alternatively, ``closing_factor`` multiplies
    the same median component size used for margin_factor (estimated once).
    Positive factors give at least one pixel for nonempty input, rounding
    down otherwise. An explicit closing takes precedence, including
    zero to disable closing. With neither option, closing is disabled.
    This changes only the
    training mask, not the labels or the component-size estimate. The gap
    width that can be bridged depends on its shape and orientation.
    closing_method="auto" uses OpenCV's ellipse for radii < 100 and
    Euclidean distance transforms for larger radii, avoiding large kernels.
    "distance" always uses a Euclidean disk; "opencv" preserves the old
    rasterized ellipse exactly. Disk and ellipse can differ at their edges.
    Block working memory is O((chunk_size + 2 * margin +
    4 * closing)**2); closing is also seamless across block borders.
    RGB input is converted with
    OpenCV's RGB-to-gray convention, as in the original implementation.
    """

    if len(label.shape) not in (2, 3) or (len(label.shape) == 3 and label.shape[2] != 3):
        raise ValueError("label must be a 2D array or an RGB image")

    if not math.isfinite(margin_factor) or margin_factor < 0:
        raise ValueError("margin_factor must be finite and nonnegative")

    if closing_method not in ("auto", "distance", "opencv"):
        raise ValueError("closing_method must be auto, distance, or opencv")

    chunk_size = operator.index(chunk_size)

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if closing is not None:
        closing = operator.index(closing)

        if closing < 0:
            raise ValueError("closing must be nonnegative")

    if closing_factor is not None and (not math.isfinite(closing_factor) or closing_factor < 0):
        raise ValueError("closing_factor must be finite and nonnegative")

    if margin is not None:
        margin = operator.index(margin)

        if margin < 0:
            raise ValueError("margin must be nonnegative")

    height, width = label.shape[:2]

    mask = np.zeros((height, width), dtype=np.uint8)

    typical_size = 0.0

    if margin is None or (closing is None and closing_factor):
        typical_size = estimate_component_size(label, chunk_size, progress=progress)

    if margin is None:
        margin = max(1, int(typical_size * margin_factor)) if typical_size and margin_factor else 0

    if closing is None:
        closing = max(1, int(typical_size * closing_factor)) if typical_size and closing_factor else 0

    closing_kernel = None
    use_distance = closing_method == "distance" or (closing_method == "auto" and closing >= 100)

    if closing and not use_distance:
        closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * closing + 1,) * 2)

    # Closing is dilation followed by erosion: each needs radius pixels.
    halo = margin + 2 * closing

    # Read a halo so that labels across block boundaries contribute too.
    for y, x in _blocks(height, width, chunk_size, progress=progress, desc="Creating training mask"):
        y1 = min(height, y + chunk_size)
        x1 = min(width, x + chunk_size)

        top, bottom = max(0, y - halo), min(height, y1 + halo)
        left, right = max(0, x - halo), min(width, x1 + halo)

        block = _binary(label[top:bottom, left:right])

        if not block.any():
            continue

        if margin and block.any():
            expanded = distance_transform_edt(block == 0) <= margin
        else:
            # Explicitly handle empty blocks: EDT otherwise measures
            # distance to implicit background outside the array.
            expanded = block

        if closing and use_distance:
            # Dilation and erosion with a Euclidean disk. EDT cost is
            # linear in block pixels rather than kernel area per pixel.
            if not expanded.all():
                expanded = distance_transform_edt(expanded == 0) <= closing

            if not expanded.all():
                expanded = distance_transform_edt(expanded) > closing

            # A full foreground block has no background to erode toward.
            # This also matches OpenCV's neutral image-border handling.
        elif closing_kernel is not None:
            expanded = cv2.morphologyEx(expanded.astype(np.uint8), cv2.MORPH_CLOSE, closing_kernel)
        
        core = expanded[y - top:y1 - top, x - left:x1 - left]
        mask[y:y1, x:x1] = core.astype(np.uint8) * 255
    
    return mask
