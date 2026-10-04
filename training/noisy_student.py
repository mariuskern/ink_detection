from pathlib import Path

from training.utils import fit

from utils import predict, postprocess_prediction, save_predictions_as_images, save_predictions_as_zarr
from utils.device import get_device
from data import build_dataset, build_dataloader, build_augmentation


def train_noisy_student(cfg, logger):
    output_dir = Path(cfg.path.output_dir)
    tmp_dir = Path(cfg.path.tmp_dir)

    for iteration in range(cfg.train.iterations):
        print(f"Starting Noisy Student iteration {iteration}")

        output_dir_iteration = output_dir / "iteration" / str(iteration)
        tmp_dir_iteration = tmp_dir / "iteration" / str(iteration)

        device = get_device(cfg)

        # Training
        result = fit(
            cfg,
            logger,
            output_dir=output_dir_iteration,
            pseudo_label_version=iteration - 1 if iteration > 0 else None,
            iteration=iteration,
            augmentation=cfg.train.student.aug,
            device=device,
        )

        if "predict" in cfg.train and cfg.train.predict is not None and len(cfg.train.predict) > 0:
            save_predictions(cfg, result.top_checkpoints, output_dir_iteration, tmp_dir_iteration, device)

        generate_pseudo_labels(cfg, result.top_checkpoints, iteration, output_dir_iteration, tmp_dir_iteration, device)


def generate_pseudo_labels(cfg, checkpoint_paths, iteration, output_dir, tmp_dir, device):
        # transform = build_augmentation(cfg.train.teacher.aug) if cfg.train.teacher.aug is not None else None
        transforms = []
        if cfg.train.teacher.aug is not None:
            for variant in cfg.train.teacher.aug.variants:
                transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, min_mask_fraction=0.0, transform=transform)
        predict_dataset = build_dataset(cfg, cfg.train.unlabeled, transform=None, split="train" if cfg.train.use_validation_mask else None)
        predict_loader = build_dataloader(cfg, predict_dataset, None)

        for prediction in predict(cfg, checkpoint_paths, predict_loader, device=device, transforms=transforms, tta_num=cfg.train.teacher.tta_num, cache_dir=tmp_dir):
            # results = predict(model, predict_loader, device=device)
            # # results = predict(model, predict_loader, device=device, tta_transform=transform, tta_num=cfg.train.refinement.tta_num)

            prediction = postprocess_prediction(cfg, prediction)

            save_predictions_as_images(
                output_dir,
                [prediction],
                save_probability=True,
                save_confidence=True,
                save_uncertainty=True,
                save_label=True,
                save_mask=True,
                save_label_not_refined=True,
                save_mask_not_refined=True,
            )

            save_predictions_as_zarr(
                Path(cfg.path.data_dir),
                [prediction],
                iteration,
                save_probability=True,
                save_label=True,
                save_mask=True,
            )


def save_predictions(cfg, checkpoint_paths, output_dir, tmp_dir, device):
        # transforms = []
        # if cfg.train.teacher.aug is not None:
        #     for variant in cfg.train.teacher.aug.variants:
        #         transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, transform=None, split="train" if cfg.train.use_validation_mask else None)
        predict_dataset = build_dataset(cfg, cfg.train.predict, transform=None, split=None)
        predict_loader = build_dataloader(cfg, predict_dataset, None)

        # for prediction in predict(model, predict_loader, device=device, transforms=transforms, tta_num=cfg.train.refinement.tta_num, cache_dir=output_dir / "tmp"):
        for prediction in predict(cfg, checkpoint_paths, predict_loader, device=device, transforms=None, tta_num=1, cache_dir=tmp_dir):
            results = [prediction]

            save_predictions_as_images(
                output_dir,
                results,
                save_probability=True,
            )
