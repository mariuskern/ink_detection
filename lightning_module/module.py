import os
from PIL import Image
from pathlib import Path

import lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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

        self.segment_buffers = {}

        self.val_img_dir = val_img_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tile, label, segment, x, y, segment_shape = batch
        output = self(tile)
        loss = self.loss(output, label)
        self.log("train/loss", loss, batch_size=self.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tile, label, segment, x, y, segment_shape = batch
        output = self(tile)
        loss = self.loss(output, label)

        probs = torch.sigmoid(output).squeeze(1).detach().to("cpu").numpy()

        for i in range(len(segment)):
            self._accumulate_prediction(
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

        for segment, buffer in self.segment_buffers.items():
            prediction = buffer["prob_sum"] / np.maximum(buffer["weight_sum"], 1e-6)

            # [0,1] -> [0,255]
            prediction_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

            path = (
                Path(self.val_img_dir)
                / f"{segment}_epoch_{self.current_epoch:04d}_prediction.png"
            )

            # Image.fromarray(prediction_uint8).save(path)
            plt.figure(figsize=(10, 10))
            plt.imshow(prediction, cmap="turbo", vmin=0, vmax=1)
            plt.axis("off")

            plt.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300
            )
            plt.close()

        self.segment_buffers.clear()

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

    def _accumulate_prediction(
        self,
        segment,
        x,
        y,
        prediction,
        segment_shape,
    ):

        def gaussian_weight(size, sigma_scale=1 / 6):
            ax = np.arange(size) - (size - 1) / 2
            xx, yy = np.meshgrid(ax, ax)
            sigma = size * sigma_scale
            g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
            return (g / g.max()).astype(np.float32)

        h, w = prediction.shape

        if segment not in self.segment_buffers:
            H, W = segment_shape[1], segment_shape[2]

            self.segment_buffers[segment] = {
                "prob_sum": np.zeros((H, W), dtype=np.float32),
                "weight_sum": np.zeros((H, W), dtype=np.float32),
            }

        buffer = self.segment_buffers[segment]
        H, W = buffer["prob_sum"].shape

        h_valid = min(h, H - y)
        w_valid = min(w, W - x)

        if h_valid <= 0 or w_valid <= 0:
            return

        prediction = prediction[:h_valid, :w_valid]
        weight = gaussian_weight(h)[:h_valid, :w_valid]
        buffer["prob_sum"][y : y + h_valid, x : x + w_valid] += prediction * weight
        buffer["weight_sum"][y : y + h_valid, x : x + w_valid] += weight
