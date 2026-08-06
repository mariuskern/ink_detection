import sys
import os
from PIL import Image

from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import zarr
from tqdm import tqdm

from .utils import train_one_iteration

# print(os.getenv("CODE_DIR"))
# sys.path.append(os.getenv("CODE_DIR"))

from utils import PredictionMerger, get_tiles, save_tiles
from data import build_unlabeled_dataloader


def train_noisy_student(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    for iteration in range(cfg.train.iterations):
        print(f"Starting Noisy Student iteration {iteration}")

        output_dir_iteration = output_dir / "iteration" / str(iteration)

        if iteration == 0:
            use_unlabeled_data = False
        else:
            use_unlabeled_data = True

        # Training
        model = train_one_iteration(cfg, logger, output_dir=output_dir_iteration, use_unlabeled_data=use_unlabeled_data, aug=cfg.train.student.aug)

        # Create pseudo labels
        unlabeled_dataloader = build_unlabeled_dataloader(cfg, aug=None, stride=48)
        predict_and_save_pseudo_labels(model, unlabeled_dataloader, cfg.path.data_dir, output_dir_iteration, cfg.train.threshold, cfg.train.confidence_threshold, cfg.train.uncertainty_threshold, iteration, cfg.dataset.tile_size, cfg.dataset.stride)


def predict_and_save_pseudo_labels(model, dataloader, data_dir, output_dir, threshold, confidence_threshold, uncertainty_threshold, iteration, tile_size, stride):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_dir = Path(data_dir)

    segments_to_paths = {}
    for folder in data_dir.iterdir():
        for segment in folder.iterdir():
            if segment.is_dir():
                segments_to_paths[segment.name] = segment

    prediction_dir = output_dir / "prediction"
    uncertainty_dir = output_dir / "uncertainty"
    confidence_dir = output_dir / "confidence"
    pseudo_label_dir = output_dir / "pseudo_label"
    mask_dir = output_dir / "mask"
    
    prediction_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    pseudo_label_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    prediction_merger = PredictionMerger()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            tile, _, _, _, segment, x, y, segment_shape = batch
            
            tile = tile.to(device)
            output = model(tile)
    
            probs = torch.sigmoid(output).squeeze(1).detach().to("cpu").numpy()
    
            for i in range(len(segment)):
                prediction_merger.add(
                    segment=segment[i],
                    x=int(x[i]),
                    y=int(y[i]),
                    prediction=probs[i],
                    segment_shape=segment_shape[i].tolist(),
                )

    results = prediction_merger.get_predictions()

    for segment, data in results.items():
        prediction = data["prediction"]
        uncertainty = data["uncertainty"]
        confidence = data["confidence"]

        _save_image(prediction_dir / f"{segment}_prediction.png", prediction)
        _save_image(uncertainty_dir / f"{segment}_uncertainty.png", uncertainty)
        _save_image(confidence_dir / f"{segment}_confidence.png", confidence)

        pseudo_label = (prediction >= threshold).astype(np.uint8)
        _save_image(pseudo_label_dir / f"{segment}_pseudo_label.png", pseudo_label)
        _save_zarr_array(segments_to_paths[segment], segment, "pseudo_labels", iteration, pseudo_label)

        mask = ((confidence > confidence_threshold) & (uncertainty < uncertainty_threshold))
        _save_image(mask_dir / f"{segment}_pseudo_mask.png", mask)
        _save_zarr_array(segments_to_paths[segment], segment, "pseudo_masks", iteration, mask)

        # tiles_dir = segments_to_paths[segment] / segment / "pseudo_tiles" / str(iteration)
        # tiles_dir.mkdir(parents=True, exist_ok=True)

        # tiles = get_tiles(pseudo_label, mask, tile_size, stride, segment)

        # npy_file = tiles_dir / f"tile_size_{tile_size}_stride_{stride}.npy"
        # csv_file = tiles_dir / f"tile_size_{tile_size}_stride_{stride}.csv"
        # save_tiles(tiles, npy_file, csv_file)

    prediction_merger.clear()


def _save_image(path, image):
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    else:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(image)
    img.save(path)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap="magma", vmin=0, vmax=1)
    # plt.axis("off")
    # plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.close()

def _save_zarr_array(data_dir, segment, group_name, iteration, array):
    zarr_path = data_dir / f"{segment}.zarr"

    z = zarr.open(zarr_path, mode="a")

    if iteration == 0 and group_name in z:
        del z[group_name]

    group = z.require_group(group_name)

    # if str(iteration) in group:
    #     del group[str(iteration)]

    # group.create_array(
    #     name=str(iteration),
    #     data=array,
    #     shape=array.shape,
    #     dtype=array.dtype,
    #     chunks=True,
    #     overwrite=True,
    # )

    group.create_array(
        str(iteration),
        shape=array.shape,
        chunks="auto",
        dtype=array.dtype,
        overwrite=True
    )

    group[str(iteration)][:] = array
