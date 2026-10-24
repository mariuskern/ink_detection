"""Example: python utils/test/plot_refined_labels.py label.png --radius 5."""

import argparse
import math
import operator
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.refine_labels import refine_labels


def plot_label_refinement(label, refined, *, max_size=2000):
    """Return a before/after figure; sample before materializing large arrays.

    Call plt.show() or fig.savefig(...) on the returned figure. Nearest-neighbor
    sampling bounds preview memory, but can hide thin strokes; use a crop for
    inspecting individual edges.
    """

    import matplotlib.pyplot as plt

    if tuple(label.shape[:2]) != tuple(refined.shape):
        raise ValueError("label and refined must have matching spatial shapes")

    max_size = operator.index(max_size)

    if max_size < 1:
        raise ValueError("max_size must be positive")

    step = max(1, math.ceil(max(label.shape[:2]) / max_size))
    before = (np.asarray(label[::step, ::step]) > 0).astype(np.uint8)
    after = np.asarray(refined[::step, ::step])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, image, title in zip(axes, (before, after), ("Original label", "Refined label")):
        artist = ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    fig.colorbar(artist, ax=list(axes), shrink=0.7, label="Target value")

    return fig


def plot_label_refinement_steps(steps, *, min_value=0.5, smooth_background=False):
    # Display intermediate arrays returned by refine_labels(return_steps=True).
    import matplotlib.pyplot as plt

    radius = steps["radius"]
    x, x1, y1, y = steps["extent"]
    images = tuple(steps[name] for name in ("original", "inside_distance", "outside_distance", "refined"))
    titles = (
        "1. Original label\nBackground = 0, ink = 1",
        "2. Ink distance to background\nWhite to black",
        "3. Background distance to ink\nBlack to white",
        f"4. Refined label\nInk = {min_value:g} to 1, background = "
        + (f"0 to {1-min_value:g}" if smooth_background else "0"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True, layout="constrained")

    for index, (ax, image, title) in enumerate(zip(axes.flat, images, titles)):
        is_distance = index in (1, 2)
        artist = ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=radius + 1 if is_distance else 1,
            interpolation="nearest",
            extent=(x, x1, y1, y),
        )
        ax.set_title(title)
        fig.colorbar(artist, ax=ax, shrink=0.8, label="Distance (px)" if is_distance else "Label value")

    fig.suptitle(f"Label refinement: y={y}:{y1}, x={x}:{x1}, transition width={radius:g} px")

    return fig


def add_label_contours(fig):
    """Overlay original-label boundaries on derived plots only.

    Contours are Matplotlib artists; no source or refined array is modified.
    Use the original preview so crops and downsampling match all panels.
    """

    image_axes = [ax for ax in fig.axes if ax.images]
    original = image_axes[0].images[0]
    binary = np.asarray(original.get_array()) > 0
    rows, cols = binary.shape

    if rows < 2 or cols < 2 or not binary.any() or binary.all():
        return

    left, right, bottom, top = original.get_extent()
    xs = left + (np.arange(cols) + 0.5) * (right - left) / cols
    ys = top + (np.arange(rows) + 0.5) * (bottom - top) / rows

    for ax in image_axes[1:]:
        ax.contour(xs, ys, binary, levels=[0.5], colors="red", linewidths=0.5)


def _get_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "label",
        type=Path,
        help="Binary image or .npy array",
    )

    radii = parser.add_mutually_exclusive_group()

    radii.add_argument(
        "--radius",
        type=float,
        help="Fixed transition width in pixels",
    )

    radii.add_argument(
        "--radius-factor",
        type=float,
        default=0.2,
        help="Transition width relative to median component size (default: 0.2)",
    )

    parser.add_argument(
        "--min-value",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--smooth-background",
        action="store_true",
        help="Also soften background: 1-min-value at ink boundary, falling to 0 over the same radius",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save float32 targets as a .npy file",
    )

    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save preview instead of opening a window",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    parser.add_argument(
        "--steps",
        action="store_true",
        help="Plot distance-transform intermediate steps",
    )

    parser.add_argument(
        "--origin",
        nargs=2,
        type=int,
        metavar=("Y", "X"),
        help="Optional crop origin (default: full image)",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=2000,
        help="Maximum full-image preview dimension",
    )

    args = parser.parse_args()

    if args.output and args.output.resolve() == args.label.resolve():
        parser.error("output must differ from input")

    if args.output and args.output.suffix.lower() != ".npy":
        parser.error("output must end in .npy")

    return args


def main(args):
    Image.MAX_IMAGE_PIXELS = None

    if args.label.suffix.lower() == ".npy":
        label = np.load(args.label, mmap_mode="r")
    else:
        # Compressed images must be decoded into RAM; use .npy for disk-backed input.
        with Image.open(args.label) as image:
            label = np.array(image.convert("L"))

    result = refine_labels(
        label,
        args.radius,
        radius_factor=args.radius_factor,
        min_value=args.min_value,
        smooth_background=args.smooth_background,
        chunk_size=args.chunk_size,
        progress=not args.no_progress,
        return_steps=args.steps,
        steps_origin=args.origin,
        steps_crop_size=args.crop_size,
        steps_max_size=args.max_size,
    )
    
    refined, steps = result if args.steps else (result, None)

    if args.output:
        np.save(args.output, refined)

    import matplotlib

    if args.save_plot:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if args.steps:
        fig = plot_label_refinement_steps(
            steps,
            min_value=args.min_value,
            smooth_background=args.smooth_background,
        )
    else:
        fig = plot_label_refinement(label, refined, max_size=args.max_size)

    add_label_contours(fig)

    if args.save_plot:
        fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main(_get_arguments())
