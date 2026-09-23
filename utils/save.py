from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import zarr

from .segment_paths import get_segment_paths


def save_predictions_as_images(
    output_dir,
    predictions,
    save_probability=False,
    save_confidence=False,
    save_uncertainty=False,
    save_label=False,
    save_mask=False,
    save_label_not_refined=False,
    save_mask_not_refined=False,
    name_suffix=None,
    create_subfolders=True,
):
    for prediction in predictions:
        if save_probability:
            _save_image(
                output_dir,
                "prediction",
                prediction.segment,
                prediction.probability,
                name_suffix,
                create_subfolders
            )

        if save_confidence and prediction.confidence is not None:
            _save_image(
                output_dir,
                "confidence",
                prediction.segment,
                prediction.confidence,
                name_suffix,
                create_subfolders=True,
            )

        if save_uncertainty and prediction.uncertainty is not None:
            _save_image(
                output_dir,
                "uncertainty",
                prediction.segment,
                prediction.uncertainty,
                name_suffix,
                create_subfolders=True,
            )

        if save_label and prediction.pseudo_label is not None:
            _save_image(
                output_dir,
                "pseudo_label",
                prediction.segment,
                prediction.pseudo_label,
                name_suffix,
                create_subfolders=True,
            )

        if save_mask and prediction.pseudo_mask is not None:
            _save_image(
                output_dir,
                "pseudo_mask",
                prediction.segment,
                prediction.pseudo_mask,
                name_suffix,
                create_subfolders=True,
            )

        if save_label_not_refined and prediction.pseudo_label_not_refined is not None:
            _save_image(
                output_dir,
                "pseudo_label_not_refined",
                prediction.segment,
                prediction.pseudo_label_not_refined,
                name_suffix,
                create_subfolders=True,
            )

        if save_mask_not_refined and prediction.pseudo_mask_not_refined is not None:
            _save_image(
                output_dir,
                "pseudo_mask_not_refined",
                prediction.segment,
                prediction.pseudo_mask_not_refined,
                name_suffix,
                create_subfolders=True,
            )


# def _save_image(output_dir, name, segment, image, name_suffix, create_subfolders):
#     directory = output_dir / name if create_subfolders else output_dir
#     directory.mkdir(parents=True, exist_ok=True)

#     filename = f"{segment}_{name}{'_' + name_suffix if name_suffix is not None else ''}.png"
#     filepath = directory / filename

#     # Image.fromarray(image).save(filename)

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image, cmap="magma")
#     ax.axis("off")
#     fig.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=300)
#     plt.close(fig)


def _save_image(output_dir, name, segment, image, name_suffix, create_subfolders):
    # print(
    #     f"{name}: shape={image.shape}, "
    #     f"dtype={image.dtype}, "
    #     f"size={image.nbytes / 1024**3:.2f} GiB"
    # )

    directory = output_dir / name if create_subfolders else output_dir
    directory.mkdir(parents=True, exist_ok=True)

    filename = f"{segment}_{name}{'_' + name_suffix if name_suffix is not None else ''}.png"
    filepath = directory / filename

    image = np.asarray(image)

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    image = np.squeeze(image)

    image_min = np.float32(image.min())
    image_max = np.float32(image.max())

    if image_max > image_min:
        image = (image - image_min) / (image_max - image_min)
        image = image.astype(np.float32, copy=False)
    else:
        image = np.zeros_like(image)

    # Matplotlib-Colormap direkt auf das Array anwenden
    # rgba = plt.get_cmap("magma")(image)

    # [0, 1] -> [0, 255]
    # rgba = (rgba * 255).astype(np.uint8)

    # Alpha-Kanal entfernen -> RGB PNG
    # rgb = rgba[..., :3]

    # 256-color lookup table
    cmap = plt.get_cmap("magma", 256)
    lut = (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)

    # [0, 1] -> [0, 255]
    indices = (image * 255).astype(np.uint8)

    # Apply colormap
    rgb = lut[indices]

    # Image.fromarray(rgb, mode="RGB").save(filepath)

    image_pil = Image.fromarray(rgb, mode="RGB")
    image_pil.thumbnail((3000, 3000), Image.Resampling.LANCZOS)

    image_pil.save(filepath)


def save_predictions_as_zarr(data_dir, predictions, iteration, save_probability=False, save_confidence=False, save_uncertainty=False, save_label=False, save_mask=False, save_label_not_refined=False, save_mask_not_refined=False):
    segment_paths = get_segment_paths(data_dir)
    
    for prediction in predictions:
        if save_probability:
            _save_zarr(
                segment_paths[prediction.segment],
                "prediction",
                # prediction.segment,
                prediction.probability,
                iteration
            )

        if save_confidence and prediction.confidence is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "confidence",
                # prediction.segment,
                prediction.confidence,
                iteration
            )

        if save_uncertainty and prediction.uncertainty is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "uncertainty",
                # prediction.segment,
                prediction.uncertainty,
                iteration
            )

        if save_label and prediction.pseudo_label is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "pseudo_labels",
                # prediction.segment,
                prediction.pseudo_label,
                iteration
            )

        if save_mask and prediction.pseudo_mask is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "pseudo_masks",
                # prediction.segment,
                prediction.pseudo_mask,
                iteration
            )

        if save_label_not_refined and prediction.pseudo_label_not_refined is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "pseudo_label_not_refined",
                # prediction.segment,
                prediction.pseudo_label_not_refined,
                iteration
            )

        if save_mask_not_refined and prediction.pseudo_mask_not_refined is not None:
            _save_zarr(
                segment_paths[prediction.segment],
                "pseudo_mask_not_refined",
                # prediction.segment,
                prediction.pseudo_mask_not_refined,
                iteration
            )


def _save_zarr(segment_path, group_name, array, iteration):
    zarr_path = segment_path / f"{segment_path.name}.zarr"

    z = zarr.open(zarr_path, mode="a")

    if iteration == 0 and group_name in z:
        del z[group_name]

    group = z.require_group(group_name)

    group.create_array(
        str(iteration),
        shape=array.shape,
        chunks="auto",
        dtype=array.dtype,
        overwrite=True
    )

    group[str(iteration)][:] = array
