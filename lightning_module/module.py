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

from utils import PredictionMerger


class LitModel(pl.LightningModule):
    def __init__(self, model, loss, lr, warmup_epochs, max_epochs, weight_decay, val_img_dir, batch_size):
        super().__init__()

        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(ignore=["model", "loss"])

        self.model = model
        self.loss = loss

        self.prediction_merger = PredictionMerger()

        self.val_img_dir = val_img_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        output = self(tile)
        loss = self.loss(output, label, mask)
        self.log("train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tile, label, mask, is_pseudo, segment, x, y, segment_shape = batch
        output = self(tile)
        loss = self.loss(output, label, mask)

        probs = torch.sigmoid(output).squeeze(1).detach().to("cpu").numpy()

        for i in range(len(segment)):
            self.prediction_merger.add(
                segment=segment[i],
                x=int(x[i]),
                y=int(y[i]),
                prediction=probs[i],
                segment_shape=segment_shape[i].tolist(),
            )

        self.log("val/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if self.global_rank != 0:
            return

        results = self.prediction_merger.get_predictions()

        for segment, data in results.items():
            prediction = data["prediction"]
            path = Path(self.val_img_dir) / f"{segment}_epoch_{self.current_epoch:04d}_prediction.png"

            plt.figure(figsize=(10, 10))
            plt.imshow(prediction, cmap="magma", vmin=0, vmax=1)
            plt.axis("off")
            plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close()

        self.prediction_merger.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,  # startet bei 1% der LR
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    #     total_steps = self.trainer.estimated_stepping_batches
    #     warmup_steps = self.warmup_epochs * (total_steps // self.max_epochs)

    #     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    #         optimizer,
    #         start_factor=0.01,  # startet bei 1% der LR
    #         end_factor=1.0,
    #         total_iters=warmup_steps, # self.warmup_epochs
    #     )

    #     # cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=1e-6)
    #     cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)

    #     scheduler = torch.optim.lr_scheduler.SequentialLR(
    #         optimizer,
    #         schedulers=[warmup_scheduler, cosine_scheduler],
    #         milestones=[warmup_steps], # [self.warmup_epochs]
    #     )

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step",
    #         },
    #     }
