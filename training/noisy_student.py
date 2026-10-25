from pathlib import Path
from datetime import datetime
import copy

import torch
from torch.optim.swa_utils import AveragedModel
from omegaconf import OmegaConf
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from training.utils import fit
from training.seed import seed_everything
from utils.predict import predict
from utils.postprocess import postprocess_prediction
from utils.save import save_predictions_as_images, save_predictions_as_zarr
from utils.device import get_device
from data import build_dataset, build_dataloader, build_augmentation
from training.precision import parse_precision
from utils.load_model import load_model_from_checkpoint
from utils.teacher_checkpoints import resolve_teacher_checkpoints, save_teacher_selection


def train_noisy_student(cfg, logger):
    output_dir = Path(cfg.path.output_dir)

    tmp_dir = Path(cfg.path.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pseudo_data_dir = tmp_dir / "pseudo_data"
    pseudo_data_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = cfg.train.get("start_iteration", 0)
    teacher_checkpoint = cfg.train.get("teacher_checkpoint")

    if not 0 <= start_iteration < cfg.train.iterations:
        raise ValueError("train.start_iteration must be >= 0 and < train.iterations")
    
    if start_iteration > 0 and not teacher_checkpoint:
        raise ValueError("train.teacher_checkpoint is required when train.start_iteration > 0")
    
    if teacher_checkpoint and start_iteration == 0:
        raise ValueError("train.teacher_checkpoint requires train.start_iteration > 0")

    teacher_paths = resolve_teacher_checkpoints(teacher_checkpoint, start_iteration) if teacher_checkpoint else []

    if teacher_checkpoint:
        previous_iteration = start_iteration - 1

        teacher_paths = backup_iteration_directories(output_dir, tmp_dir, previous_iteration, teacher_paths)

        print(f"\n{'=' * 64}\nNoisy Student | Resume: restore teacher from iteration {previous_iteration}\n{'=' * 64}")

        cfg_previous_iteration = _get_config_for_iteration(cfg, previous_iteration)

        generate_pseudo_labels(
            cfg_previous_iteration,
            teacher_paths,
            previous_iteration,
            output_dir / "iteration" / str(previous_iteration),
            tmp_dir / "iteration" / str(previous_iteration),
            get_device(cfg_previous_iteration),
            pseudo_data_dir,
        )

    for iteration in range(start_iteration, cfg.train.iterations):
        print(f"\n{'=' * 64}\nNoisy Student | Iteration {iteration} (of 0..{cfg.train.iterations - 1})\n{'=' * 64}")
        cfg_iteration = _get_config_for_iteration(cfg, iteration)

        if cfg_iteration.train.seed is not None:
            iteration_seed = cfg_iteration.train.seed + iteration
            seed_everything(iteration_seed)
            print(f"  Seed:      {iteration_seed}")

        output_dir_iteration = output_dir / "iteration" / str(iteration)
        tmp_dir_iteration = tmp_dir / "iteration" / str(iteration)

        if start_iteration > 0:
            backup_iteration_directories(output_dir, tmp_dir, iteration)

        device = get_device(cfg_iteration)

        # Training
        result = fit(
            cfg_iteration,
            logger,
            output_dir=output_dir_iteration,
            pseudo_label_version=iteration - 1 if iteration > 0 else None,
            iteration=iteration,
            augmentation=cfg_iteration.train.student.aug,
            device=device,
            pseudo_data_dir=pseudo_data_dir,
        )

        checkpoint_paths = get_checkpoint_paths(cfg_iteration, result, device)
        print(f"\n  Teacher: {len(checkpoint_paths)} checkpoint(s)")
        for path in checkpoint_paths:
            print(f"    {path}")

        if "predict" in cfg_iteration.train and cfg_iteration.train.predict is not None and len(cfg_iteration.train.predict) > 0:
            save_predictions(cfg_iteration, checkpoint_paths, output_dir_iteration, tmp_dir_iteration, device)

        generate_pseudo_labels(cfg_iteration, checkpoint_paths, iteration, output_dir_iteration, tmp_dir_iteration, device, pseudo_data_dir)
        print(f"  Iteration {iteration} complete.")


def backup_iteration_directories(output_dir, tmp_dir, iteration, checkpoint_path=None):
    """Preserve iteration outputs and caches, relocating the teacher path if needed."""

    multiple = isinstance(checkpoint_path, list)
    paths = checkpoint_path if multiple else ([checkpoint_path] if checkpoint_path is not None else [])
    paths = [path.resolve() for path in paths]

    directories = {}

    for root in (output_dir, tmp_dir):
        directory = (root / "iteration" / str(iteration)).resolve()
        directory.relative_to(root.resolve())
        directories[directory] = None

    for directory in directories:
        if directory.exists():
            backup = directory.with_name(f"{directory.name}_backup_{datetime.now():%Y%m%d_%H%M%S_%f}")

            directory.rename(backup)

            paths = [backup / path.relative_to(directory) if path.is_relative_to(directory) else path for path in paths]

            print(f"  Backup:    {backup}")
    
    return paths if multiple else (paths[0] if paths else None)


def generate_pseudo_labels(cfg, checkpoint_paths, iteration, output_dir, tmp_dir, device, pseudo_data_dir):
        print(f"\n  Pseudo labels | teacher iteration {iteration} | {len(checkpoint_paths)} checkpoint(s)")
        # Reset here so a resumed run reproduces the same TTA and loader randomness.
        if cfg.train.seed is not None:
            seed_everything(cfg.train.seed + iteration)

        # transform = build_augmentation(cfg.train.teacher.aug) if cfg.train.teacher.aug is not None else None
        transforms = []
        if cfg.train.teacher.aug is not None:
            for variant in cfg.train.teacher.aug.variants:
                transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, min_mask_fraction=0.0, transform=transform)
        predict_dataset = build_dataset(cfg, cfg.train.unlabeled, use_as_unlabeled=True, pseudo_label_version=None, transform=None, split="train" if cfg.train.use_validation_mask else None, use_mask_refinement=False, use_label_refinement=False)
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
                save_confidence_kde=True,
                save_uncertainty_kde=True,
            )

            save_predictions_as_zarr(
                pseudo_data_dir,
                [prediction],
                iteration,
                save_probability=True,
                save_label=True,
                save_mask=True,
            )

        print(f"  Pseudo labels saved: {pseudo_data_dir} (version {iteration})")


def save_predictions(cfg, checkpoint_paths, output_dir, tmp_dir, device):
        print(f"\n  Export predictions | {len(checkpoint_paths)} checkpoint(s)")
        # transforms = []
        # if cfg.train.teacher.aug is not None:
        #     for variant in cfg.train.teacher.aug.variants:
        #         transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, transform=None, split="train" if cfg.train.use_validation_mask else None)
        predict_dataset = build_dataset(cfg, cfg.train.predict, use_as_unlabeled=True, pseudo_label_version=None, transform=None, split=None, use_mask_refinement=False, use_label_refinement=False)
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

    save_teacher_selection(checkpoint_paths)
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
        # Construction copies weights but leaves n_averaged at zero.
        averaged_model.update_parameters(loaded_model)

        del loaded_model

    torch.save(
        {
            "state_dict": averaged_model.module.state_dict(),
            "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        },
        output_path,
    )

    del averaged_model

    # return output_path


def _get_config_for_iteration(cfg, iteration):
    cfg_iteration = copy.deepcopy(cfg)

    if iteration < 0:
        raise ValueError("iteration must be non-negative")

    models = cfg.train.get("model_schedule")

    if models is None:
        print(f"  Model:     {cfg_iteration.model.name} (fixed configuration)")
        return cfg_iteration

    if isinstance(models, str):
        models = [models]

    if not models:
        raise ValueError("train.model_schedule must contain at least one model config name")

    name = models[min(iteration, len(models) - 1)]

    # Replay only explicit model-field overrides, preserving Hydra's syntax.
    # The model group itself is selected by the iteration schedule.
    model_overrides = []
    if HydraConfig.initialized():
        model_overrides = [override for override in HydraConfig.get().overrides.task if override.split("=", 1)[0].lstrip("+~").startswith("model.")]

    model_cfg = compose(
        config_name=None,
        overrides=[f"+model={name}", *model_overrides],
    ).model

    with open_dict(cfg_iteration):
        cfg_iteration.model = model_cfg

    print(f"  Model:     {name} (architecture: {model_cfg.name})")
    if model_overrides:
        print(f"  Overrides: {', '.join(model_overrides)}")

    return cfg_iteration
