from pathlib import Path

from .utils import train_one_iteration

from utils import predict, postprocess_predictions, save_predictions_as_images, save_predictions_as_zarr
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
        model = train_one_iteration(cfg, logger, output_dir=output_dir_iteration, use_unlabeled_data=use_unlabeled_data, aug=cfg.train.student.aug, iteration=iteration)


        # Create pseudo labels

        unlabeled_dataloader = build_unlabeled_dataloader(cfg, aug=None, stride=cfg.dataset.stride)

        results = predict(model, unlabeled_dataloader, device=cfg.train.devices)
        results = postprocess_predictions(
            results,
            cfg.train.threshold,
            cfg.train.confidence_threshold,
            cfg.train.uncertainty_threshold,
            use_confidence_mask=cfg.train.use_confidence_mask,
            use_uncertainty_mask=cfg.train.use_uncertainty_mask,
        )

        save_predictions_as_images(output_dir_iteration, results, save_probability=True, save_confidence=True, save_uncertainty=True, save_label=True, save_mask=True)
        save_predictions_as_zarr(
            Path(cfg.path.data_dir),
            results,
            iteration,
            save_probability=True,
            save_label=True,
            save_mask=True,
        )
