"""Blockwise soft targets for binary ink labels."""

import math
import operator

import numpy as np
from scipy.ndimage import distance_transform_edt

from utils.create_training_mask import _binary, _blocks, estimate_component_size


def resolve_refinement_radius(label, radius=None, *, radius_factor=0.2, chunk_size=2048, progress=True):
    """Resolve pixel radius; explicit radius takes precedence over the factor.

    Factor mode uses max(1, int(median_component_size * radius_factor)),
    matching the training-mask size estimate. Empty labels use radius 1.
    Components crossing internal block boundaries are excluded from estimation;
    increase chunk_size if no complete component can be measured.
    """

    if not math.isfinite(radius_factor) or radius_factor <= 0:
        raise ValueError("radius_factor must be finite and positive")

    if radius is not None:
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")

        return float(radius)

    size = estimate_component_size(label, chunk_size, progress=progress)

    return float(max(1, int(size * radius_factor)))


def _distance_ramp(distance, radius):
    return np.clip((distance - 1) / radius, 0, 1).astype(np.float32)


def _refinement_values(block, core, radius, min_value, smooth_background, *, return_distances=False):
    """Compute both displayed ramp and targets using the same block halo."""

    distance = distance_transform_edt(np.pad(block, 1))[1:-1, 1:-1]
    foreground = block[core].astype(bool)
    ramp = _distance_ramp(distance[core], radius)
    soft = (min_value + (1 - min_value) * ramp) * foreground
    outside_distance = None

    if (smooth_background or return_distances) and block.any():
        # No padding with foreground: outside the image is not an ink source.
        outside_distance = distance_transform_edt(block == 0)[core]

    if smooth_background and outside_distance is not None:
        outside_ramp = 1 - _distance_ramp(outside_distance, radius)
        # Display a continuous, symmetric transition: background 0,
        # both sides of the boundary 0.5, deep foreground 1.
        ramp = np.where(foreground, 0.5 + 0.5 * ramp, 0.5 * outside_ramp)
        soft = np.where(foreground, soft, (1 - min_value) * outside_ramp)

    if return_distances:
        # No ink in the halo means every background distance is saturated.
        if outside_distance is None:
            outside_distance = np.full(foreground.shape, radius + 1)

        inside = np.minimum(distance[core], radius + 1).astype(np.float32)
        outside = np.where(foreground, 0, np.minimum(outside_distance, radius + 1)).astype(np.float32)

        return ramp, soft, inside, outside

    return ramp, soft


def refine_labels(
    label,
    radius=None,
    *,
    radius_factor=0.2,
    min_value=0.5,
    chunk_size=2048,
    progress=True,
    smooth_background=False,
    return_steps=False,
    steps_origin=None,
    steps_crop_size=512,
    steps_max_size=2000,
):
    """Soften positive labels inward, optionally also the surrounding background.

    Without radius, use radius_factor times the estimated median component
    size (rounded down, at least 1 pixel), like create_training_mask.

    Any input value > 0 is foreground (RGB uses the training-mask grayscale
    convention). Returns float32 targets, not a 0/255 image. For foreground,
    value = min_value + (1-min_value) * clip((distance-1)/radius, 0, 1).
    Thus boundary pixels have min_value and pixels at distance >= radius+1
    have value 1. Thin strokes may never reach 1. min_value=0 allows the
    range [0, 1], but also makes the outermost positive pixels zero.

    With smooth_background=True, background pixels adjacent to ink receive
    1-min_value, decreasing to zero at distance >= radius+1 from ink.
    The same radius/factor applies to both sides. At min_value=0.5 this is
    symmetric; values below 0.5 invert the ordering at the boundary.
    Without this option background remains zero. Outside-image pixels are
    never treated as ink sources for background smoothing.

    This is a fixed-width distance ramp, not skeleton or per-component
    normalization. Image borders are treated as background outside the image.
    A halo of ceil(radius+1) makes results independent of chunk_size.
    Runtime is linear in processed pixels; temporary memory is bounded by
    O((chunk_size + 2*ceil(radius+1))**2). Large radii increase overlap work.

    Input can be a sliceable array (e.g. memmap/Zarr). The returned float32
    array has shape label.shape[:2] and is allocated in RAM (4 bytes/pixel).

    With return_steps=True, return (refined, steps). steps contains sampled original,
    inside_distance, outside_distance, refined, radius and extent for plotting.
    Sampling is bounded by steps_max_size; steps_origin selects a full-resolution
    square crop of steps_crop_size. Distances are clipped to radius+1.
    """

    if len(label.shape) not in (2, 3) or (len(label.shape) == 3 and label.shape[2] != 3):
        raise ValueError("label must be a 2D array or an RGB image")

    if not math.isfinite(min_value) or not 0 <= min_value <= 1:
        raise ValueError("min_value must be between 0 and 1")

    chunk_size = operator.index(chunk_size)

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    radius = resolve_refinement_radius(
        label,
        radius,
        radius_factor=radius_factor,
        chunk_size=chunk_size,
        progress=progress,
    )

    shape = tuple(label.shape[:2])

    refined = np.empty(shape, dtype=np.float32)

    height, width = shape
    halo = math.ceil(radius + 1)
    steps = None

    if return_steps:
        max_size = operator.index(steps_max_size)
        crop_size = operator.index(steps_crop_size)

        if max_size < 1 or crop_size < 1:
            raise ValueError("steps_max_size and steps_crop_size must be positive")

        y0, x0 = (0, 0) if steps_origin is None else map(operator.index, steps_origin)

        if not (0 <= y0 < height and 0 <= x0 < width):
            raise ValueError("steps_origin must lie inside the image")

        end_y, end_x = (height, width) if steps_origin is None else (min(height, y0 + crop_size), min(width, x0 + crop_size))
        
        step = max(1, math.ceil(max(height, width) / max_size)) if steps_origin is None else 1
        preview_shape = (math.ceil((end_y - y0) / step), math.ceil((end_x - x0) / step))
        steps = {
            name: np.empty(preview_shape, dtype=np.float32)
            for name in ("original", "inside_distance", "outside_distance", "refined")
        }

        steps.update(
            radius=radius,
            extent=(x0, end_x, end_y, y0),
        )

    for y, x in _blocks(height, width, chunk_size, progress=progress, desc="Refining labels"):
        y1, x1 = min(height, y + chunk_size), min(width, x + chunk_size)

        top, bottom = max(0, y - halo), min(height, y1 + halo)
        left, right = max(0, x - halo), min(width, x1 + halo)

        block = _binary(label[top:bottom, left:right])

        core = np.s_[y - top : y1 - top, x - left : x1 - left]

        result = _refinement_values(
            block,
            core,
            radius,
            min_value,
            smooth_background,
            return_distances=return_steps,
        )
        
        values = result[1]
        refined[y:y1, x:x1] = values

        if return_steps:
            first_y = max(y, y0)
            first_x = max(x, x0)
            first_y += (y0 - first_y) % step
            first_x += (x0 - first_x) % step
            last_y, last_x = min(y1, end_y), min(x1, end_x)

            if first_y >= last_y or first_x >= last_x:
                continue

            sample = np.s_[first_y - y : last_y - y : step, first_x - x : last_x - x : step]
            target = np.s_[
                (first_y - y0) // step : math.ceil((last_y - y0) / step),
                (first_x - x0) // step : math.ceil((last_x - x0) / step),
            ]

            steps["original"][target] = block[core][sample]
            steps["refined"][target] = values[sample]
            steps["inside_distance"][target] = result[2][sample]
            steps["outside_distance"][target] = result[3][sample]

    return (refined, steps) if return_steps else refined
