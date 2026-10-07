from pathlib import Path

import torch
from torch.optim.swa_utils import AveragedModel

from training.utils import fit
from utils import predict, postprocess_prediction, save_predictions_as_images, save_predictions_as_zarr
from utils.device import get_device
from data import build_dataset, build_dataloader, build_augmentation
from training.precision import parse_precision
from utils.load_model import load_model_from_checkpoint


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

        checkpoint_paths = get_checkpoint_paths(cfg, result, device)

        if "predict" in cfg.train and cfg.train.predict is not None and len(cfg.train.predict) > 0:
            save_predictions(cfg, checkpoint_paths, output_dir_iteration, tmp_dir_iteration, device)

        generate_pseudo_labels(cfg, checkpoint_paths, iteration, output_dir_iteration, tmp_dir_iteration, device)


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


def get_checkpoint_paths(cfg, result, device):
    checkpoint_paths = []

    # if cfg.train.teacher.use_best and result.best_checkpoint:
    #     checkpoint_paths.append(result.best_checkpoint)

    if cfg.train.teacher.use_top_k and result.top_checkpoints:
        match cfg.train.teacher.top_k_mode:
            case "average":
                top_k_average_path = result.top_checkpoints[0].parent / "top_k_average.pt"
                save_average_checkpoint(cfg, result.top_checkpoints, device=device, output_path=top_k_average_path)
                checkpoint_paths.append(top_k_average_path)
            case "individual":
                checkpoint_paths.extend(result.top_checkpoints)
            case _:
                raise ValueError(f"Invalid top_k_mode: {cfg.train.teacher.top_k_mode}")

    if cfg.train.teacher.use_swa and result.swa_checkpoints:
        match cfg.train.teacher.swa_mode:
            case "average":
                swa_average_path = result.swa_checkpoints[0].parent / "swa_average.pt"
                save_average_checkpoint(cfg, result.swa_checkpoints, device=device, output_path=swa_average_path)
                checkpoint_paths.append(swa_average_path)
            case "individual":
                checkpoint_paths.extend(result.swa_checkpoints)
            case _:
                raise ValueError(f"Invalid swa_mode: {cfg.train.teacher.swa_mode}")

    return checkpoint_paths


def save_average_checkpoint(cfg, checkpoint_paths: list[Path], device, output_path: Path) -> None:
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided.")

    _, _, model_dtype = parse_precision(cfg.train.precision)

    averaged_model = None

    for checkpoint_path in checkpoint_paths:
        loaded_model = load_model_from_checkpoint(cfg=cfg, checkpoint_path=checkpoint_path, model_dtype=model_dtype, device=device)

        if averaged_model is None:
            averaged_model = AveragedModel(loaded_model)
        else:
            averaged_model.update_parameters(loaded_model)

        del loaded_model

    torch.save(
        {
            "state_dict": averaged_model.module.state_dict(),
        },
        output_path,
    )

    del averaged_model

    # return output_path
