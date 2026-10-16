# NOTE: Additional information about the model can be found in `resnet3d_reference.py`.


"""HuggingFace-style wrapper around the ResNet3D-50 + 2-D U-Net ink-detection model.

This file is **self-contained** — vendored ResNet3D-50 (Hara et al., 2018)
inline with the decoder so a downloader only needs `transformers` and
`torch`.  Loadable via:

    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        "<user>/<repo>", trust_remote_code=True
    )

Input: float32 tensor of shape `(B, 1, D, H, W)` or `(B, D, H, W)`,
       where D = 62, H = W = 256 (intensity already z-score normalised).
Output: `ModelOutput(logits=<sigmoid logits, shape (B, 1, H/4, W/4)>)`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List


# =============================================================================
# Vendored ResNet3D-50 (Hara, Kataoka & Satoh, 2018)
# =============================================================================
def _conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


def _conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = _conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = _conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class _ResNet3D(nn.Module):
    """ResNet3D-50 backbone returning the 4 intermediate feature maps."""

    def __init__(self, n_input_channels=1, block_inplanes=(64, 128, 256, 512),
                 layers=(3, 4, 6, 3), conv1_t_size=7, conv1_t_stride=1):
        super().__init__()
        self.in_planes = block_inplanes[0]
        self.conv1 = nn.Conv3d(
            n_input_channels, self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.layer1 = self._make_layer(block_inplanes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block_inplanes[3], layers[3], stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * _Bottleneck.expansion:
            downsample = nn.Sequential(
                _conv1x1x1(self.in_planes,
                           planes * _Bottleneck.expansion, stride),
                nn.BatchNorm3d(planes * _Bottleneck.expansion),
            )
        layers = [_Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * _Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(_Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x) -> List[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


# =============================================================================
# 2-D U-Net decoder
# =============================================================================
class _Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1],
                          encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True),
            )
            for i in range(1, len(encoder_dims))
        ])
        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode='bilinear')

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2,
                                 mode='bilinear')
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            feature_maps[i - 1] = self.convs[i - 1](f)
        return self.up(self.logit(feature_maps[0]))


class VesuviusResNet3D_HF(nn.Module):
    """Vesuvius Challenge ink-detection model.

    Pipeline:
        1.  3-D volume `(B, 1, D, H, W)` enters a ResNet3D-50 backbone.
        2.  Each of the 4 stages is collapsed along the z (depth) axis
            with `torch.max` -> 2-D feature pyramid.
        3.  A small 2-D U-Net decoder upsamples coarse-to-fine with
            concatenated skip connections.
        4.  A 1x1 conv head produces 1 logit channel.
    """

    base_model_prefix = "inkdetection"

    def __init__(
        self,
        in_channels=1,
        backbone_depth=50,
        backbone_channels=(256, 512, 1024, 2048),
        decoder_upscale=1,
    ):
        super().__init__()

        layers_map = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3)
        }

        if backbone_depth not in layers_map:
            raise ValueError(
                f"Unsupported backbone_depth={backbone_depth}; "
                "expected one of 50, 101, 152."
            )

        self.backbone = _ResNet3D(
            n_input_channels=in_channels,
            block_inplanes=(64, 128, 256, 512),
            layers=layers_map[backbone_depth],
        )

        self.decoder = _Decoder(
            encoder_dims=list(backbone_channels),
            upscale=decoder_upscale,
        )

    def forward(self, pixel_values):
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)

        if pixel_values.ndim != 5:
            raise ValueError(
                f"pixel_values must be 4-D (B, D, H, W) or "
                f"5-D (B, 1, D, H, W); "
                f"got shape {tuple(pixel_values.shape)}"
            )

        feats = self.backbone(pixel_values)
        pooled = [torch.max(f, dim=2)[0] for f in feats]
        logits = self.decoder(pooled)

        return logits

    @torch.no_grad()
    def predict_probability(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """Convenience: sigmoid probabilities at quarter resolution."""
        return torch.sigmoid(self(pixel_values))
