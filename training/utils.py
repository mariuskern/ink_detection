"""
fit()
│
├── build_model()
├── build_loss()
├── build_dataset()
├── build_dataloader()
├── build_optimizer()
├── build_scheduler()
│
└── for epoch:
    │
    ├── train_one_epoch()
    │   ├── forward
    │   ├── labeled loss
    │   ├── pseudo loss
    │   ├── backward
    │   ├── gradient clipping
    │   ├── optimizer.step()
    │   ├── scheduler.step()
    │   └── metrics
    │
    ├── validate()
    │   ├── forward
    │   ├── validation loss
    │   ├── tile dice
    │   ├── tile IoU
    │   └── PredictionMerger.add()
    │
    └── validation_epoch_end()
        ├── PredictionMerger.get_binary_metrics()
        ├── segment dice
        ├── segment IoU
        ├── save_predictions_as_images()
        └── PredictionMerger.clear()
"""


from pathlib import Path
from dataclasses import dataclass
import math
import gc
import os

import torch
# import torch.nn.functional as F
from tqdm import tqdm

from data.builder import build_train_dataset, build_dataset, build_sampler, build_dataloader
from models import build_model
from omegaconf import OmegaConf
from losses import build_loss
from utils.prediction_merger import PredictionMerger
from utils.save import save_predictions_as_images
from utils.evaluation import calculate_dice, calculate_iou
from utils.device import get_device
from .precision import parse_precision, autocast_context
from .checkpoint import CheckpointManager
from .logging import LoggingManager
from utils.interpolate import interpolate, interpolate_reverse


@dataclass(frozen=True)
class TrainingResult:
    # model: torch.nn.Module
    best_checkpoint: Path | None
    best_score: float | None

    # top_models: list[torch.nn.Module]
    top_checkpoints: list[Path]
    swa_checkpoints: list[Path]
    # last_model: torch.nn.Module | None
    last_checkpoint: Path | None


def fit(cfg, logger, output_dir, *, pseudo_label_version=None, iteration=None, augmentation=None, device=None, pseudo_data_dir=None):
    output_dir = Path(output_dir)

    checkpoint_dir = output_dir / "checkpoints"
    val_image_dir = output_dir / "val_images"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    if device is None:
        device = get_device(cfg)
    print(f"  Device:    {device} | precision: {cfg.train.precision} | compile: {cfg.train.compile}")

    # ------------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------------

    autocast_dtype, use_grad_scaler, model_dtype = parse_precision(cfg.train.precision)

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_grad_scaler)

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = build_model(cfg)
    # model = model.to(device)
    if model_dtype is not None:
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device)
    if cfg.train.compile:
        from torch._functorch import config as functorch_config

        # Backward runs outside autocast in train_one_epoch. Keep this setting
        # active through the first forward, when compilation actually happens.
        functorch_config.backward_pass_autocast = "off"
        model = torch.compile(model, mode=cfg.train.compile_mode)

    # ------------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------------

    loss_fn = build_loss(cfg)

    # ------------------------------------------------------------------------
    # Dataset / DataLoader
    # ------------------------------------------------------------------------

    use_unlabeled_data = iteration is not None and iteration > 0

    train_dataset = build_train_dataset(cfg, use_unlabeled_data=use_unlabeled_data, pseudo_label_version=pseudo_label_version, augmentation=augmentation, split="train" if cfg.train.use_validation_mask else None, pseudo_data_dir=pseudo_data_dir, use_mask_refinement=True, use_label_refinement=True)
    # val_dataset = build_dataset(cfg, cfg.train.val, min_mask_fraction=0.0, split="val" if cfg.train.use_validation_mask else None)
    val_dataset = build_dataset(cfg, cfg.train.val, split="val" if cfg.train.use_validation_mask else None, use_mask_refinement=True, use_label_refinement=False)

    train_sampler = build_sampler(cfg, train_dataset, use_unlabeled_data=use_unlabeled_data)
    
    train_loader = build_dataloader(cfg, train_dataset, train_sampler)
    val_loader = build_dataloader(cfg, val_dataset, None)

    print(f"  Samples:   train={len(train_dataset):,} | validation={len(val_dataset):,}")
    print(f"  Output:    {output_dir}")

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # ------------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------------

    accumulation_steps = cfg.train.gradient_accumulation_steps

    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)

    total_steps = cfg.train.max_epochs * optimizer_steps_per_epoch
    steps_per_epoch = max(optimizer_steps_per_epoch, 1)

    requested_warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
    warmup_steps = min(requested_warmup_steps, max(total_steps - 1, 0))

    # total_steps = cfg.train.max_epochs * len(train_loader)
    # steps_per_epoch = max(total_steps // cfg.train.max_epochs, 1)

    # requested_warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
    # warmup_steps = min(requested_warmup_steps, max(total_steps - 1, 0))

    if warmup_steps == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=1e-6,
        )
    else:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    # ------------------------------------------------------------------------
    # Prediction merger
    # ------------------------------------------------------------------------

    prediction_merger = PredictionMerger()

    # ------------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------------

    checkpoint_manager = CheckpointManager(
        dirpath=checkpoint_dir,
        monitor=cfg.train.checkpoint.monitor,
        mode=cfg.train.checkpoint.mode,
        save_top_k=cfg.train.checkpoint.save_top_k,
        save_last=cfg.train.checkpoint.save_last,
        save_last_every_epoch=cfg.train.checkpoint.save_last_every_epoch,
        filename_template=cfg.model.name + "_{epoch:03d}_{val_loss:.3f}",
        every_n_epochs=cfg.train.checkpoint.checkpoint_every_n_epochs,
        swa_save_k=cfg.train.checkpoint.swa_save_k,
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
    )

    logging_manager = LoggingManager(
        logger=logger,
        iteration=str(iteration) if iteration is not None else None
    )

    if iteration:
        logging_manager.log_once("train_debug", "iteration", iteration)

    # ------------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------------

    for epoch in range(cfg.train.max_epochs):
        print(f"\n  Epoch {epoch + 1}/{cfg.train.max_epochs} | learning rate: {optimizer.param_groups[0]['lr']:.3e}")

        # ====================================================================
        # TRAIN
        # ====================================================================

        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            cfg=cfg,
            epoch=epoch,
            # iteration=iteration,
            # logger=logger,
            logger=logging_manager,
            pseudo_labels_used=True if pseudo_label_version is not None else False,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            accumulation_steps=accumulation_steps
        )

        # ====================================================================
        # VALIDATION
        # ====================================================================

        image_interval = cfg.train.val_image_every_n_epochs
        segment_metrics_interval = cfg.train.val_segment_metrics_every_n_epochs

        export_predictions = image_interval > 0 and (epoch + 1) % image_interval == 0
        compute_segment_metrics = segment_metrics_interval > 0 and (epoch + 1) % segment_metrics_interval == 0

        val_metrics = validate(
            model=model,
            loss_fn=loss_fn,
            val_loader=val_loader,
            device=device,
            cfg=cfg,
            epoch=epoch,
            prediction_merger=prediction_merger,
            export_predictions=export_predictions,
            compute_segment_metrics=compute_segment_metrics,
            val_image_dir=val_image_dir,
            # logger=logger,
            logger=logging_manager,
            pseudo_labels_used=True if pseudo_label_version is not None else False,
            autocast_dtype=autocast_dtype
        )

        metrics = logging_manager.compute()

        # ====================================================================
        # Print epoch summary
        # ====================================================================

        print(f"  Loss       train: {metrics['train/loss']:.6f} | validation: {metrics['val/loss']:.6f}")

        # ====================================================================
        # ModelCheckpoint equivalent
        # ====================================================================

        # checkpoint_metrics = {
        #     **train_metrics,
        #     **val_metrics,
        # }

        checkpoint_manager.step(
            epoch=epoch,
            metrics=metrics,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # ====================================================================
        # Logging
        # ====================================================================

        logging_manager.log_epoch()
    

    # ------------------------------------------------------------------------
    # Load top-k models and last model
    # ------------------------------------------------------------------------

    checkpoint_manager.clear()

    top_checkpoints = checkpoint_manager.top_k_model_paths
    swa_checkpoints = checkpoint_manager.swa_model_paths
    last_checkpoint = checkpoint_manager.last_model_path
    best_checkpoint = checkpoint_manager.best_model_path
    best_score = checkpoint_manager.best_model_score

    # def load_model_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    #     checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    #     # Build a fresh, uncompiled model
    #     loaded_model = build_model(cfg)

    #     if model_dtype is not None:
    #         loaded_model = loaded_model.to(device=device, dtype=model_dtype)
    #     else:
    #         loaded_model = loaded_model.to(device)

    #     state_dict = checkpoint["state_dict"]

    #     # Remove torch.compile prefix if necessary
    #     if any(key.startswith("_orig_mod.") for key in state_dict):
    #         state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    #     loaded_model.load_state_dict(state_dict)
    #     loaded_model.eval()

    #     del checkpoint
    #     del state_dict

    #     return loaded_model

    # Load the top-k models in best-first order
    # top_models = [load_model_from_checkpoint(path) for path in top_checkpoints]

    # Load the last model independently
    # last_model = (load_model_from_checkpoint(last_checkpoint) if last_checkpoint is not None else None)

    # Backward compatibility: result.model is still the best model
    # if top_models:
    #     best_model = top_models[0]
    # elif last_model is not None:
    #     best_model = last_model
    # else:
    #     best_model = model
    #     best_model.eval()

    # Clean up training resources
    del optimizer
    del scheduler
    del scaler

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return TrainingResult(
        # model=best_model,
        best_checkpoint=best_checkpoint,
        best_score=best_score,
        # top_models=top_models,
        top_checkpoints=top_checkpoints,
        swa_checkpoints=swa_checkpoints,
        # last_model=last_model,
        last_checkpoint=last_checkpoint,
    )

    # # ------------------------------------------------------------------------
    # # Load best model
    # # ------------------------------------------------------------------------

    # best_checkpoint = checkpoint_manager.best_model_path
    # best_score = checkpoint_manager.best_model_score

    # if best_checkpoint is not None:
    #     checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)

    #     # Build a fresh, uncompiled model
    #     inference_model = build_model(cfg)

    #     if model_dtype is not None:
    #         inference_model = inference_model.to(device=device, dtype=model_dtype)
    #     else:
    #         inference_model = inference_model.to(device)

    #     # inference_model.load_state_dict(checkpoint["state_dict"])

    #     state_dict = checkpoint["state_dict"]

    #     is_compiled_checkpoint = any(key.startswith("_orig_mod.") for key in state_dict.keys())

    #     if is_compiled_checkpoint:
    #         state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}

    #     inference_model.load_state_dict(state_dict)
    #     inference_model.eval()

    #     del checkpoint
    #     del state_dict

    #     del model
    #     del optimizer
    #     del scheduler
    #     del scaler
    #     # torch.cuda.empty_cache()
    # else:
    #     # Fallback if no checkpoint exists
    #     inference_model = model
    #     inference_model.eval()

    # gc.collect()
    # torch.cuda.empty_cache()

    # return TrainingResult(
    #     model=inference_model,
    #     best_checkpoint=best_checkpoint,
    #     best_score=best_score,
    # )

    # best_checkpoint = checkpoint_manager.best_model_path
    # best_score = checkpoint_manager.best_model_score

    # if best_checkpoint is not None:
    #     checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    #     model.load_state_dict(checkpoint["state_dict"])
    #     model.to(device)

    # return TrainingResult(
    #     model=model,
    #     best_checkpoint=best_checkpoint,
    #     best_score=best_score,
    # )


# ============================================================================
# Training epoch
# ============================================================================

def train_one_epoch(
    *,
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_loader,
    device,
    cfg,
    epoch,
    # iteration,
    logger,
    pseudo_labels_used,
    autocast_dtype,
    scaler,
    accumulation_steps
):
    """
    Equivalent to SupervisedModule.training_step(), executed over one epoch.
    """

    model.train()

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="  Train", dynamic_ncols=True, leave=False)):

        # --------------------------------------------------------------------
        # Move batch to device
        # --------------------------------------------------------------------

        tile = batch["tile"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        pseudo_label = batch["pseudo_label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pseudo_mask = batch["pseudo_mask"].to(device, non_blocking=True)
        has_label = batch["has_label"].to(device)
        has_pseudo_label = batch["has_pseudo_label"].to(device)
        has_pseudo_mask = batch["has_pseudo_mask"].to(device)

        if (has_label & has_pseudo_label).any():
            raise RuntimeError("Sample cannot be both labeled and pseudo-labeled.")

        if (~(has_label | has_pseudo_label)).any():
            raise RuntimeError("Sample has neither label nor pseudo-label.")

        # --------------------------------------------------------------------
        # Interpolation
        # --------------------------------------------------------------------

        # interpolate_labels = cfg.model.name == "timesformer"

        # if interpolate_labels:
        #     label = interpolate(label, cfg.model.patch_size)
        #     pseudo_label = interpolate(pseudo_label, cfg.model.patch_size)
        #     mask = interpolate(mask, cfg.model.patch_size)
        #     pseudo_mask = interpolate(pseudo_mask, cfg.model.patch_size)

        # --------------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------------

        # optimizer.zero_grad(set_to_none=True)

        with autocast_context(device.type, autocast_dtype):
            output = model(tile)

            needs_resize = label.shape[-2:] != output.shape[-2:]

            if needs_resize:
                label = interpolate(label, output.shape[-2:])
                pseudo_label = interpolate(pseudo_label, output.shape[-2:])
                mask = interpolate(mask, output.shape[-2:])
                pseudo_mask = interpolate(pseudo_mask, output.shape[-2:])

            # --------------------------------------------------------------------
            # Labeled loss
            # --------------------------------------------------------------------

            if has_label.any():
                labeled_loss = loss_fn(output[has_label], label[has_label], mask[has_label])
            else:
                labeled_loss = torch.tensor(0.0, device=device)

            # --------------------------------------------------------------------
            # Pseudo loss
            # --------------------------------------------------------------------

            if has_pseudo_label.any():
                pseudo_loss = loss_fn(output[has_pseudo_label], pseudo_label[has_pseudo_label], pseudo_mask[has_pseudo_label])
            else:
                pseudo_loss = torch.tensor(0.0, device=device)

            # --------------------------------------------------------------------
            # Pseudo loss weight
            # --------------------------------------------------------------------

            # if pseudo_labels_used:
            #     pseudo_weight = get_pseudo_loss_weight(cfg, epoch)
            # else:
            #     pseudo_weight = 0.0

            # --------------------------------------------------------------------
            # Total loss
            # --------------------------------------------------------------------

            # loss = labeled_loss + pseudo_weight * pseudo_loss

            combined_label = torch.empty_like(label)
            combined_mask = torch.empty_like(mask)

            combined_label[has_label] = label[has_label]
            combined_label[has_pseudo_label] = pseudo_label[has_pseudo_label]

            combined_mask[has_label] = mask[has_label]
            combined_mask[has_pseudo_label] = pseudo_mask[has_pseudo_label]

            # sample_weight = torch.where(has_label, torch.ones_like(has_label, dtype=torch.float32), torch.full_like(has_label, pseudo_weight, dtype=torch.float32))
            # loss = loss_fn(output, combined_label, combined_mask, sample_weight=sample_weight)
            
            loss = loss_fn(output, combined_label, combined_mask)

        # --------------------------------------------------------------------
        # Backward
        # --------------------------------------------------------------------

        # loss.backward()
        scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        # scaler.scale(loss).backward()

        should_step = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader)

        if should_step:
            # --------------------------------------------------------------------
            # Gradient clipping
            # --------------------------------------------------------------------

            gradient_clip_val = cfg.train.gradient_clip_val

            if gradient_clip_val is not None:
                scaler.unscale_(optimizer)

                algorithm = str(cfg.train.gradient_clip_algorithm).lower()

                if algorithm == "norm":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                elif algorithm == "value":
                    torch.nn.utils.clip_grad_value_(model.parameters(), gradient_clip_val)
                else:
                    raise ValueError("Unknown gradient clipping algorithm: " f"{algorithm!r}")

            # --------------------------------------------------------------------
            # Optimizer
            # --------------------------------------------------------------------

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            # --------------------------------------------------------------------
            # Scheduler
            #
            # Lightning:
            #
            #     interval="step"
            #
            # therefore this is intentionally here and NOT after the epoch.
            # --------------------------------------------------------------------

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------------------
        # Training metrics
        # --------------------------------------------------------------------

        if has_label.any():
            num_labeled = has_label.sum().item()

            tile_dice = calculate_dice(
                output[has_label],
                label[has_label],
                mask[has_label],
                threshold=cfg.train.threshold,
            )

            tile_iou = calculate_iou(
                output[has_label],
                label[has_label],
                mask[has_label],
                threshold=cfg.train.threshold,
            )
        else:
            num_labeled = 0
            tile_dice = torch.tensor(0.0, device=device)
            tile_iou = torch.tensor(0.0, device=device)
        
        logger.log_dict(
            "train",
            {
                "tile_dice": tile_dice,
                "tile_iou": tile_iou,
            },
            batch_size=num_labeled,
            on_step=True,
            on_epoch=True,
        )

        if has_pseudo_label.any():
            num_pseudo_labeled = has_pseudo_label.sum().item()

            pseudo_tile_dice = calculate_dice(
                output[has_pseudo_label],
                pseudo_label[has_pseudo_label],
                pseudo_mask[has_pseudo_label],
                threshold=cfg.train.threshold,
            )

            pseudo_tile_iou = calculate_iou(
                output[has_pseudo_label],
                pseudo_label[has_pseudo_label],
                pseudo_mask[has_pseudo_label],
                threshold=cfg.train.threshold,
            )
        else:
            num_pseudo_labeled = 0
            pseudo_tile_dice = torch.tensor(0.0, device=device)
            pseudo_tile_iou = torch.tensor(0.0, device=device)

        if pseudo_labels_used:
            logger.log_dict(
                "train",
                {
                    "pseudo_tile_dice": pseudo_tile_dice,
                    "pseudo_tile_iou": pseudo_tile_iou,
                },
                batch_size=num_pseudo_labeled,
                on_step=True,
                on_epoch=True,
            )

        # --------------------------------------------------------------------
        # Combined metrics
        # --------------------------------------------------------------------

        if pseudo_labels_used:
            logits_combined = torch.cat([output[has_label], output[has_pseudo_label]])
            targets_combined = torch.cat([label[has_label], pseudo_label[has_pseudo_label]])
            masks_combined = torch.cat([mask[has_label],pseudo_mask[has_pseudo_label]])

            combined_tile_dice = calculate_dice(
                logits_combined,
                targets_combined,
                masks_combined,
                threshold=cfg.train.threshold,
            )

            combined_tile_iou = calculate_iou(
                logits_combined,
                targets_combined,
                masks_combined,
                threshold=cfg.train.threshold,
            )

            logger.log_dict(
                "train",
                {
                    "combined_tile_dice": combined_tile_dice,
                    "combined_tile_iou": combined_tile_iou,
                },
                batch_size=cfg.dataloader.batch_size,
                on_step=True,
                on_epoch=True,
            )

        # --------------------------------------------------------------------
        # Loss metrics
        # --------------------------------------------------------------------

        logger.log_dict(
            "train",
            {
                "loss": loss,
            },
            batch_size=cfg.dataloader.batch_size,
            on_step=True,
            on_epoch=True,
        )

        if pseudo_labels_used:
            logger.log_dict(
                "train",
                {
                    "labeled_loss": labeled_loss,
                    "pseudo_loss": pseudo_loss,
                    # "pseudo_weight": pseudo_weight,
                },
                batch_size=cfg.dataloader.batch_size,
                on_step=True,
                on_epoch=True,
            )

        # --------------------------------------------------------------------
        # Debug values
        # --------------------------------------------------------------------

        num_samples_with_pseudo_label = has_pseudo_label.sum()

        if has_pseudo_label.any():
            pseudo_label_mean = pseudo_label[has_pseudo_label].mean()
        else:
            pseudo_label_mean = torch.tensor(0, device=pseudo_label.device, dtype=pseudo_label.dtype)

        num_samples_with_pseudo_mask = has_pseudo_mask.sum()

        if has_pseudo_mask.any():
            pseudo_mask_mean = pseudo_mask[has_pseudo_mask].mean()
        else:
            pseudo_mask_mean = torch.tensor(0, device=pseudo_mask.device, dtype=pseudo_mask.dtype)

        if pseudo_labels_used:
            logger.log_dict(
                "train_debug",
                {
                    "num_samples_with_pseudo_label": num_samples_with_pseudo_label,
                    "pseudo_label_mean": pseudo_label_mean,
                    "num_samples_with_pseudo_mask": num_samples_with_pseudo_mask,
                    "pseudo_mask_mean": pseudo_mask_mean,
                },
                batch_size=cfg.dataloader.batch_size,
                on_step=True,
                on_epoch=True,
            )

        def count_positive_tiles(labels, masks, available, threshold):
            valid = masks > 0
            valid_pixels = valid.flatten(1).sum(dim=1)
            positive_fraction = (labels * valid).flatten(1).sum(dim=1) / valid_pixels.clamp_min(1)

            return (available & (valid_pixels > 0) & (positive_fraction > threshold)).sum().item()


        num_positive_labeled_tiles = count_positive_tiles(label, mask, has_label, cfg.sampler.pos_threshold)
        num_positive_pseudo_tiles = count_positive_tiles(pseudo_label, pseudo_mask, has_pseudo_label, cfg.sampler.pos_threshold)

        logger.log_dict(
            "train_debug",
            {
                "num_positive_tiles": num_positive_labeled_tiles + num_positive_pseudo_tiles,
                "num_positive_labeled_tiles": num_positive_labeled_tiles,
                "num_positive_pseudo_tiles": num_positive_pseudo_tiles,
            },
            batch_size=cfg.dataloader.batch_size,
            on_step=True,
            on_epoch=True,
        )

        # --------------------------------------------------------------------
        # LR monitor
        # --------------------------------------------------------------------

        current_lr = optimizer.param_groups[0]["lr"]

        logger.log(
            "train",
            "lr",
            current_lr,
            batch_size=cfg.dataloader.batch_size,
            on_step=True,
            on_epoch=False,
        )

    return logger.compute()


# ============================================================================
# Validation
# ============================================================================


@torch.no_grad()
def validate(
    *,
    model,
    loss_fn,
    val_loader,
    device,
    cfg,
    epoch,
    prediction_merger,
    export_predictions,
    compute_segment_metrics,
    val_image_dir,
    logger,
    pseudo_labels_used,
    autocast_dtype,
):
    """
    Equivalent to:

        validation_step()
        +
        on_validation_epoch_end()
    """

    model.eval()

    # interpolate_labels = cfg.model.name == "timesformer"

    # ------------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------------

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="  Validate", dynamic_ncols=True, leave=False)):
        # --------------------------------------------------------------------
        # Batch
        # --------------------------------------------------------------------

        tile = batch["tile"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        has_label = batch["has_label"].to(device)

        segment = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]

        # --------------------------------------------------------------------
        # Interpolation
        # --------------------------------------------------------------------

        # if interpolate_labels:
        #     label = interpolate(label, cfg.model.patch_size)
        #     mask = interpolate(mask, cfg.model.patch_size)

        # --------------------------------------------------------------------
        # Forward
        # --------------------------------------------------------------------

        with autocast_context(device.type, autocast_dtype):
            output = model(tile)

            label_original = label
            mask_original = mask

            needs_resize = label.shape[-2:] != output.shape[-2:]
            
            if needs_resize:
                label = interpolate(label, output.shape[-2:])
                mask = interpolate(mask, output.shape[-2:])

            # --------------------------------------------------------------------
            # Loss + tile metrics
            # --------------------------------------------------------------------

            if has_label.any():
                labeled_output = output[has_label]
                labeled_label = label[has_label]
                labeled_mask = mask[has_label]

                loss = loss_fn(labeled_output, labeled_label, labeled_mask)

                last_tile_dice = calculate_dice(
                    labeled_output,
                    labeled_label,
                    labeled_mask,
                    threshold=cfg.train.threshold,
                )

                last_tile_iou = calculate_iou(
                    labeled_output,
                    labeled_label,
                    labeled_mask,
                    threshold=cfg.train.threshold,
                )

                num_labeled = has_label.sum().item()

                # tile_dice_sum += last_tile_dice.detach().item() * num_labeled
                # tile_dice_weight += num_labeled
                # tile_iou_sum += last_tile_iou.detach().item() * num_labeled
                # tile_iou_weight += num_labeled
            else:
                loss = output.sum() * 0.0
                last_tile_dice = output.sum() * 0.0
                last_tile_iou = output.sum() * 0.0
                num_labeled = 0

            logger.log_dict(
                "val",
                {
                    "loss": loss,
                    "tile_dice": last_tile_dice,
                    "tile_iou": last_tile_iou,
                },
                batch_size=num_labeled,
                on_step=True,
                on_epoch=True,
            )

        # --------------------------------------------------------------------
        # Prediction merger
        # --------------------------------------------------------------------

        if export_predictions or compute_segment_metrics:
            if needs_resize:
                output_for_export = interpolate_reverse(output, tile.shape[-2:])
                # label_for_export = interpolate_reverse(label, tile.shape[-2:])
                # mask_for_export = interpolate_reverse(mask, tile.shape[-2:])
            else:
                output_for_export = output
                # label_for_export = label
                # mask_for_export = mask

            probs = torch.sigmoid(output_for_export).squeeze(1).detach().float().cpu().numpy()
            targets = label_original.squeeze(1).detach().float().cpu().numpy()
            valid_masks = mask_original.squeeze(1).detach().float().cpu().numpy()
            has_label_cpu = has_label.detach().cpu().numpy()

            for i in range(len(segment)):
                if not has_label_cpu[i]:
                    continue

                prediction_merger.add(
                    segment=segment[i],
                    x=int(x[i]),
                    y=int(y[i]),
                    prediction=probs[i],
                    shape=shape[i].tolist(),
                    target=targets[i],
                    valid_mask=valid_masks[i],
                )

    # ------------------------------------------------------------------------
    # PredictionMerger equivalent to on_validation_epoch_end()
    # ------------------------------------------------------------------------

    if compute_segment_metrics:
        metrics = prediction_merger.get_binary_metrics(threshold=cfg.train.threshold)

        if metrics is not None:
            log_metrics = {
                "segment_dice": metrics["dice"],
                "segment_iou": metrics["iou"],
            }

            for segment, values in metrics["per_segment"].items():
                log_metrics[f"per_segment/{segment}/dice"] = values["dice"]
                log_metrics[f"per_segment/{segment}/iou"] = values["iou"]

            logger.log_dict_once("val", log_metrics)

    if export_predictions:
        for prediction in prediction_merger.iter_predictions():
            save_predictions_as_images(
                val_image_dir,
                [prediction],
                save_probability=True,
                name_suffix=(f"epoch_{epoch:04d}"),
                create_subfolders=False,
            )

        # results = prediction_merger.get_predictions()

        # save_predictions_as_images(
        #     val_image_dir,
        #     results,
        #     save_probability=True,
        #     name_suffix=(f"epoch_{epoch:04d}"),
        #     create_subfolders=False,
        # )

    # ------------------------------------------------------------------------
    # Important:
    #
    # Lightning clears the merger at the end of every validation epoch.
    # ------------------------------------------------------------------------

    prediction_merger.clear()

    return logger.compute()


# ============================================================================
# Scheduler / loss helpers
# ============================================================================


# def get_pseudo_loss_weight(cfg, epoch):
#     """
#     Exact equivalent of:

#         SupervisedModule._pseudo_loss_weight()
#     """

#     weight = cfg.train.pseudo_loss_weight

#     if cfg.train.pseudo_loss_rampup_epochs <= 0:
#         return weight

#     factor = min(epoch / cfg.train.pseudo_loss_rampup_epochs, 1.0)

#     return weight * factor


# def interpolate(x, target_size):
#     """
#     Exact equivalent of SupervisedModule.interpolate().
#     """

#     _, _, height, width = x.shape

#     # x = F.interpolate(x, (height // interpolation_size[0], width // interpolation_size[1]), mode="nearest")
#     x = F.interpolate(x.float(), size=target_size, mode="nearest")

#     return x


# def interpolate_reverse(x, target_size):
#     """
#     Exact equivalent of SupervisedModule.interpolate_reverse().
#     """

#     # x = F.interpolate(x.float(), scale_factor=interpolation_size, mode="bilinear").squeeze(0)
#     x = F.interpolate(x.float(), size=target_size, mode="bilinear")

#     return x


# ============================================================================
# Device
# ============================================================================


# def get_device(cfg):
#     accelerator = str(cfg.train.accelerator).lower()

#     if accelerator in {"gpu", "cuda"}:
#         if not torch.cuda.is_available():
#             raise RuntimeError("GPU/CUDA was requested, " "but CUDA is not available.")
#         return torch.device("cuda")
    
#     if accelerator == "cpu":
#         return torch.device("cpu")

#     if torch.cuda.is_available():
#         return torch.device("cuda")

#     return torch.device("cpu")

def configure_determinism(*, deterministic: bool, benchmark: bool):
    if deterministic and benchmark:
        raise ValueError("deterministic=True and benchmark=True are incompatible for reproducible training.")

    if deterministic:
        # Must be configured before the first cuBLAS operation.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    torch.use_deterministic_algorithms(deterministic)
