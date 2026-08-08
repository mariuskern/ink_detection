import os
import sys
from PIL import Image
from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# print("moduel:")
# print(os.getenv("CODE_DIR"))
# sys.path.append(os.getenv("CODE_DIR"))

from utils import PredictionMerger, save_predictions_as_images


def _require_finite(name, value, batch_idx):
    """Fail at the first non-finite tensor instead of corrupting a run."""
    if value is not None and not torch.isfinite(value).all():
        non_finite = (~torch.isfinite(value)).sum().item()
        raise RuntimeError(
            f"Non-finite values in {name} at batch {batch_idx}: "
            f"{non_finite} of {value.numel()} values are NaN or Inf."
        )


class LitModel(pl.LightningModule):
    def __init__(self, model, loss, lr, warmup_epochs, max_epochs, weight_decay, val_img_dir, batch_size, iteration=None):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.iteration = iteration

        # self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss

        self.prediction_merger = PredictionMerger()

        self.val_img_dir = val_img_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]
        _require_finite("training input", tile, batch_idx)
        _require_finite("training label", label, batch_idx)
        _require_finite("training mask", mask, batch_idx)
        output = self(tile)
        _require_finite("training model output", output, batch_idx)
        loss = self.loss(output, label, mask)
        _require_finite("training loss", loss, batch_idx)
        # self.log(f"train/{str(self.iteration) + '_' if self.iteration is not None else ''}loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        if self.iteration is not None:
            self.log(f"train/{str(self.iteration)}_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(f"train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
            
        return loss

    def validation_step(self, batch, batch_idx):
        # tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        tile = batch["tile"]
        label = batch["label"]
        mask = batch["mask"]
        segment = batch["segment"]
        x = batch["x"]
        y = batch["y"]
        shape = batch["shape"]
        _require_finite("validation input", tile, batch_idx)
        _require_finite("validation label", label, batch_idx)
        _require_finite("validation mask", mask, batch_idx)
        output = self(tile)
        _require_finite("validation model output", output, batch_idx)
        loss = self.loss(output, label, mask)
        _require_finite("validation loss", loss, batch_idx)

        probs = torch.sigmoid(output).squeeze(1).detach().float().to("cpu").numpy()

        for i in range(len(segment)):
            self.prediction_merger.add(
                segment=segment[i],
                x=int(x[i]),
                y=int(y[i]),
                prediction=probs[i],
                shape=shape[i].tolist(),
            )

        # self.log(f"val/{str(self.iteration) + '_' if self.iteration is not None else ''}loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        if self.iteration is not None:
            self.log(f"val/{str(self.iteration)}_loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"val/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log(f"val/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        results = self.prediction_merger.get_predictions()
        save_predictions_as_images(self.val_img_dir, results, save_probability=True, name_suffix=f"epoch_{self.current_epoch:04d}", create_subfolders=False)

        # for segment, data in results.items():
        #     prediction = data["prediction"]
        #     path = Path(self.val_img_dir) / f"{segment}_epoch_{self.current_epoch:04d}_prediction.png"

        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(prediction, cmap="magma", vmin=0, vmax=1)
        #     plt.axis("off")
        #     plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        #     plt.close()

        self.prediction_merger.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = max(total_steps // self.max_epochs, 1)
        requested_warmup_steps = self.warmup_epochs * steps_per_epoch
        # A warmup cannot consume the entire run; otherwise the subsequent
        # cosine scheduler receives a non-positive T_max (e.g. debug runs).
        warmup_steps = min(requested_warmup_steps, max(total_steps - 1, 0))

        if warmup_steps == 0:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(total_steps, 1), eta_min=1e-6
                    ),
                    "interval": "step",
                },
            }

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # startet bei 1% der LR
            end_factor=1.0,
            total_iters=warmup_steps, # self.warmup_epochs
        )

        # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps], # [self.warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
