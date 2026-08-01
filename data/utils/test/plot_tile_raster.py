import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _get_arguments():
    parser = argparse.ArgumentParser(description="Create raster plot from tile index.")

    parser.add_argument(
        "input",
        type=Path,
        help="Path to tile index .npy file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path",
    )

    return parser.parse_args()


def _create_raster(index):
    """
    Convert tile index into a 2D raster.

    Each pixel represents one tile and contains
    the positive label fraction.
    """

    x = index["x"]
    y = index["y"]
    values = index["positive_fraction"]

    # determine tile grid spacing
    x_positions = np.unique(x)
    y_positions = np.unique(y)

    x_map = {value: i for i, value in enumerate(x_positions)}

    y_map = {value: i for i, value in enumerate(y_positions)}

    raster = np.full(
        (
            len(y_positions),
            len(x_positions),
        ),
        np.nan,
        dtype=np.float32,
    )

    for xi, yi, value in zip(
        x,
        y,
        values,
    ):
        raster[
            y_map[yi],
            x_map[xi],
        ] = value

    return raster


def _plot_raster(index, output=None):
    raster = _create_raster(index)

    fig, ax = plt.subplots(figsize=(10, 10))

    image = ax.imshow(
        raster,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    ax.set_title("Tile ink fraction")
    ax.set_xlabel("Tile x index")
    ax.set_ylabel("Tile y index")

    # plt.colorbar(
    #     image,
    #     ax=ax,
    #     label="Positive label fraction",
    # )

    plt.tight_layout()

    if output:
        plt.savefig(
            output,
            dpi=300,
            bbox_inches="tight",
        )
    else:
        plt.show()


def main(args):
    index = np.load(args.input)

    _plot_raster(index, args.output)


if __name__ == "__main__":
    main(_get_arguments())
