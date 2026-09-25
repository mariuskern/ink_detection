import os
import sys

import torch
from torchvision import tv_tensors

import hydra
from omegaconf import DictConfig

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.getenv("CODE_DIR"))

from data import build_augmentation


# ============================================================
# Hilfsfunktionen
# ============================================================

def make_circle(H, W, cy, cx, r):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2).float()


def make_rectangle(H, W, y0, y1, x0, x1):
    mask = torch.zeros((H, W), dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def make_triangle(H, W, points):
    """
    Einfaches Dreieck über baryzentrische Koordinaten.
    points = [(y1,x1), (y2,x2), (y3,x3)]
    """
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    p1 = torch.tensor(points[0], dtype=torch.float32)
    p2 = torch.tensor(points[1], dtype=torch.float32)
    p3 = torch.tensor(points[2], dtype=torch.float32)

    px = xx.float()
    py = yy.float()

    denominator = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])

    a = ((p2[1] - p3[1]) * (py - p3[0]) + (p3[0] - p2[0]) * (px - p3[1])) / denominator
    b = ((p3[1] - p1[1]) * (py - p3[0]) + (p1[0] - p3[0]) * (px - p3[1])) / denominator
    c = 1 - a - b

    return ((a >= 0) & (b >= 0) & (c >= 0)).float()


# ============================================================
# Synthetisches "CT"-Volumen erzeugen
# ============================================================

def create_test_data(H=256, W=256, D=8):
    """
    Erzeugt:

        image: (1, D, H, W)

        mask_circle:    (1, H, W)
        mask_rectangle: (1, H, W)
        mask_triangle:  (1, H, W)

    Die Objekte befinden sich auf unterschiedlichen D-Slices.
    """

    image = torch.zeros((1, D, H, W), dtype=torch.float32)

    mask_circle = torch.zeros((1, H, W), dtype=torch.float32)
    mask_rectangle = torch.zeros((1, H, W), dtype=torch.float32)
    mask_triangle = torch.zeros((1, H, W), dtype=torch.float32)

    # --------------------------------------------------------
    # Hintergrund
    # --------------------------------------------------------

    image[:] = 0.05

    # --------------------------------------------------------
    # Kreis
    # --------------------------------------------------------

    circle = make_circle(H, W, cy=75, cx=75, r=35)

    mask_circle[0] = circle

    # Kreis liegt auf Slice 2-4
    for d in [2, 3, 4]:
        image[0, d] += circle * 0.7

    # --------------------------------------------------------
    # Rechteck
    # --------------------------------------------------------

    rectangle = make_rectangle(H, W, y0=140, y1=210, x0=40, x1=120)

    mask_rectangle[0] = rectangle

    # Rechteck liegt auf Slice 3-5
    for d in [3, 4, 5]:
        image[0, d] += rectangle * 0.5

    # --------------------------------------------------------
    # Dreieck
    # --------------------------------------------------------

    triangle = make_triangle( H, W, points=[(50, 170), (120, 220), (120, 150)])

    mask_triangle[0] = triangle

    # Dreieck liegt auf Slice 5-7
    for d in [5, 6, 7]:
        image[0, d] += triangle * 0.9

    # --------------------------------------------------------
    # Etwas "CT-artiger" Hintergrund
    # --------------------------------------------------------

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

    center_y = H / 2
    center_x = W / 2

    radius = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    body = (radius < H * 0.45).float()

    image += body.unsqueeze(0).unsqueeze(1) * 0.05

    image = image.clamp(0, 1)

    return image, mask_circle, mask_rectangle, mask_triangle


def plot_result(image_before, masks_before, image_after, masks_after):
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt

    """
    Plot all depth layers and masks in a single figure.

    Layout
    ------

    Columns:
        D=0, D=1, ..., D=D-1

    Rows:
        Original image
        Transformed image

        Original mask 1
        Transformed mask 1

        Original mask 2
        Transformed mask 2

        ...

    This makes it easy to detect whether depth layers
    have been swapped/reordered.

    Image:
        (C, D, H, W)

    Mask:
        (C, H, W)
    """

    # ============================================================
    # Convert to numpy
    # ============================================================

    image_before = image_before.detach().cpu().numpy()
    image_after = image_after.detach().cpu().numpy()

    masks_before = [mask.detach().cpu().numpy() for mask in masks_before]

    masks_after = [ mask.detach().cpu().numpy() for mask in masks_after]

    # ============================================================
    # Dimensions
    # ============================================================

    C, D, H, W = image_before.shape

    n_masks = len(masks_before)

    if len(masks_after) != n_masks:
        raise ValueError("Number of masks before and after transformation must be identical.")

    # ============================================================
    # Number of rows
    #
    # 2 rows per mask:
    #
    # Image original
    # Image transformed
    #
    # Mask 1 original
    # Mask 1 transformed
    #
    # Mask 2 original
    # Mask 2 transformed
    #
    # ...
    # ============================================================

    n_rows = 2 + 2 * n_masks

    # ============================================================
    # Figure
    # ============================================================

    fig, axes = plt.subplots(n_rows, D, figsize=(3.5 * D, 3.5 * n_rows), squeeze=False)

    # ============================================================
    # Helper function for consistent axes
    # ============================================================

    def setup_axis(ax, title):
        ax.set_title(title, fontsize=6)

        # ax.set_xlabel("X [pixel]", fontsize=6)
        # ax.set_ylabel("Y [pixel]", fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")

        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

        ax.set_aspect("equal")

    # ============================================================
    # Depth layers
    # ============================================================

    for d in range(D):
        # --------------------------------------------------------
        # Original image
        # --------------------------------------------------------

        ax = axes[0, d]
        ax.imshow(image_before[0, d], cmap="gray", origin="upper", extent=[0, W, H, 0])
        setup_axis(ax, f"Original D = {d}")

        # --------------------------------------------------------
        # Transformed image
        # --------------------------------------------------------

        ax = axes[1, d]
        ax.imshow(image_after[0, d],cmap="gray", origin="upper", extent=[0, W, H, 0])
        setup_axis(ax, f"Transformed D = {d}")

    # ============================================================
    # Masks
    # ============================================================

    for mask_idx in range(n_masks):
        original_row = 2 + mask_idx * 2
        transformed_row = original_row + 1

        for d in range(D):
            # ----------------------------------------------------
            # Original mask
            # ----------------------------------------------------

            ax = axes[original_row, d]
            mask = masks_before[mask_idx][0]
            ax.imshow(mask, cmap="gray", origin="upper", extent=[0, W, H, 0], vmin=0, vmax=1, interpolation="nearest")
            setup_axis(ax, f"Mask {mask_idx + 1} - Original D = {d}")

            # ----------------------------------------------------
            # Transformed mask
            # ----------------------------------------------------

            ax = axes[transformed_row, d]
            mask = masks_after[mask_idx][0]
            ax.imshow(mask, cmap="gray", origin="upper", extent=[0, W, H, 0], vmin=0, vmax=1, interpolation="nearest")
            setup_axis(ax, f"Mask {mask_idx + 1} - Transformed D = {d}")

    # ============================================================
    # Row labels
    # ============================================================

    axes[0, 0].set_ylabel("Image Original Y [pixel]", fontsize=6)

    axes[1, 0].set_ylabel("Image Transformed Y [pixel]", fontsize=6)

    for mask_idx in range(n_masks):
        original_row = 2 + mask_idx * 2
        transformed_row = original_row + 1

        axes[original_row, 0].set_ylabel(f"Mask {mask_idx + 1} Original Y [pixel]", fontsize=6)
        axes[transformed_row, 0].set_ylabel(f"Mask {mask_idx + 1} Transformed Y [pixel]", fontsize=6)

    # ============================================================
    # Figure title
    # ============================================================

    fig.suptitle("Comparison of depth layers and masks", fontsize=10, fontweight="bold")

    # ============================================================
    # Layout
    # ============================================================

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.show()


# ============================================================
# MAIN
# ============================================================

@hydra.main(version_base=None, config_path="../../conf/aug", config_name="test")
def main(cfg: DictConfig):
    # --------------------------------------------------------
    # Daten erzeugen
    # --------------------------------------------------------

    image, mask_circle, mask_rectangle, mask_triangle = create_test_data()

    # image = tv_tensors.Image(image)
    # mask_circle = tv_tensors.Mask(mask_circle)
    # mask_rectangle = tv_tensors.Mask(mask_rectangle)
    # mask_triangle = tv_tensors.Mask(mask_triangle)
    
    masks = [mask_circle, mask_rectangle, mask_triangle]

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    transform = build_augmentation(cfg)

    # --------------------------------------------------------
    # Transformation
    #
    # Wichtig:
    #
    # image:
    #     (C,D,H,W)
    #
    # mask:
    #     (C,H,W)
    #
    # --------------------------------------------------------

    image_transformed, mask_circle_transformed, mask_rectangle_transformed, mask_triangle_transformed = transform(image, *masks)

    masks_transformed = [mask_circle_transformed, mask_rectangle_transformed, mask_triangle_transformed]

    # --------------------------------------------------------
    # Shapes ausgeben
    # --------------------------------------------------------

    print("Image:")
    print("  before:", image.shape)
    print("  after: ", image_transformed.shape)

    for i, (before, after) in enumerate(zip(masks, masks_transformed)):
        print(f"Mask {i}:")
        print("  before:", before.shape)
        print("  after: ", after.shape)

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_result(
        image_before=image,
        masks_before=masks,
        image_after=image_transformed,
        masks_after=masks_transformed,
        # slice_idx=4,
    )


if __name__ == "__main__":
    main()