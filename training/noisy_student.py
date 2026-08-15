from pathlib import Path

from training.supervised import fit

from utils import predict, postprocess_predictions, save_predictions_as_images, save_predictions_as_zarr
from data import build_dataset, build_dataloader, prepare_pseudo_tile_manifests


def train_noisy_student(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    for iteration in range(cfg.train.iterations):
        print(f"Starting Noisy Student iteration {iteration}")

        output_dir_iteration = output_dir / "iteration" / str(iteration)

        # Training
        result = fit(
            cfg,
            logger,
            output_dir=output_dir_iteration,
            pseudo_label_version=iteration - 1 if iteration > 0 else None,
            augmentation=cfg.train.student.aug,
        )

        generate_pseudo_labels(cfg, result.model, iteration, output_dir_iteration)


def generate_pseudo_labels(cfg, model, iteration, output_dir):
        predict_dataset = build_dataset(cfg, cfg.train.unlabeled, min_mask_fraction=0.0)
        predict_loader = build_dataloader(cfg, predict_dataset, None)

        results = predict(model, predict_loader, device=cfg.train.devices)
        results = postprocess_predictions(
            results,
            cfg.train.threshold,
            cfg.train.confidence_threshold,
            cfg.train.uncertainty_threshold,
            use_confidence_mask=cfg.train.use_confidence_mask,
            use_uncertainty_mask=cfg.train.use_uncertainty_mask,
        )

        save_predictions_as_images(
            output_dir,
            results,
            save_probability=True,
            save_confidence=True,
            save_uncertainty=True,
            save_label=True,
            save_mask=True
        )

        save_predictions_as_zarr(
            Path(cfg.path.data_dir),
            results,
            iteration,
            save_probability=True,
            save_label=True,
            save_mask=True,
        )

        prepare_pseudo_tile_manifests(
            cfg.path.data_dir,
            cfg.train.unlabeled,
            version=iteration,
            tile_size=cfg.dataset.tile_size,
            stride=cfg.dataset.stride,
            target=cfg.train.pseudo_target,
        )
