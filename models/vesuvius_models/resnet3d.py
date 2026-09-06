# The original model was loaded using AutoModel from the provided checkpoint,
# while the model was independently reimplemented based on the original
# repository implementation. Functional equivalence was validated by loading
# the same checkpoint weights into both models and verifying identical outputs
# for the same inputs.


import torch
import torch.nn as nn

from .challenge_code_e583fb6.resnetall import generate_model
from .challenge_code_e583fb6.model import Decoder, replace_batchnorm_with_groupnorm


class VesuviusResNet3D(nn.Module):
    def __init__(
        self,
        model_depth=50,
        n_input_channels=1,
        forward_features=True,
        n_classes=1039,
        with_norm=False,
        norm="batch",
        group_norm_groups=32,
        upscale="auto",
    ):
        super().__init__()

        self.with_norm = with_norm

        self.backbone = generate_model(
            model_depth=model_depth,
            n_input_channels=n_input_channels,
            forward_features=forward_features,
            n_classes=n_classes,
        )

        if norm == "group":
            replace_batchnorm_with_groupnorm(self.backbone, desired_groups=group_norm_groups)

        dummy_x = torch.rand(1, 1, 20, 256, 256)

        was_training = self.backbone.training
        try:
            self.backbone.eval()
            with torch.no_grad():
                # encoder_dims = [x.size(1) for x in self.backbone(torch.rand(1, 1, 20, 256, 256))]
                dummy_features = self.backbone(dummy_x)
        finally:
            if was_training:
                self.backbone.train()

        encoder_dims = [x.size(1) for x in dummy_features]

        if upscale == "auto":
            decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups,)

            with torch.no_grad():
                dummy_features_2d = [torch.max(x, dim=2)[0]for x in dummy_features]
                dummy_output = decoder(dummy_features_2d)

            input_size = dummy_x.shape[-2:]
            output_size = dummy_output.shape[-2:]

            if input_size[0] % output_size[0] != 0 or input_size[1] % output_size[1] != 0:
                raise ValueError(f"Cannot automatically determine integer upscale factor: decoder output is {output_size}, input is {input_size}.")

            upscale_h = input_size[0] // output_size[0]
            upscale_w = input_size[1] // output_size[1]

            if upscale_h != upscale_w:
                raise ValueError(f"Non-uniform upscale factor detected: height={upscale_h}, width={upscale_w}.")

            upscale = upscale_h

        if not isinstance(upscale, int) or upscale < 1:
            raise ValueError(f"`upscale` must be a positive integer or 'auto', got {upscale!r}.")
        
        self.decoder = Decoder(encoder_dims=encoder_dims, upscale=upscale, norm=norm, group_norm_groups=group_norm_groups)

    def forward(self, x):
        # x: [B, C, T, H, W]

        if self.with_norm:
            x = self.norm(x)

        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)

        return pred_mask
