# NOTE: This model was adapted with AI assistance from the Vesuvius Challenge
# implementation at:
# https://github.com/ScrollPrize/villa/commit/5479453a76f0db39a8d657434d3d4c4c517f7245
# It provides a standalone, configurable implementation whose default arguments
# reproduce the original ResNet3D encoder and 2D ink-detection decoder architecture.
# Comparisons in resnet3d_test.py use the following reference models:
# - VesuviusResNet3D_GitHub, based on ScrollPrize/villa at:
#   https://github.com/ScrollPrize/villa/commit/e583fb67468f483fadd73d52e06f0ab0fe5ba813
# - VesuviusResNet3D_HF, adapted from the Hugging Face implementation at:
#   https://huggingface.co/scrollprize/PHerc.1667-iteration-5
# Both reference models are provided through resnet3d_reference.py.
# A checkpoint from the original model was successfully loaded. With these weights
# and identical inputs, both models produced exactly identical outputs
# (maximum and mean absolute difference: 0).
# This equivalence was verified for the tested configuration and inputs.


from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 forward_features=False,
                 stage_strides=(1, 2, 2, 2),
                 stem_spatial_kernel=7,
                 stem_spatial_stride=2,
                 pool_kernel=(1, 3, 3),
                 pool_stride=(1, 2, 2),
                 pool_padding=(0, 1, 1),
                 bn_eps=1e-5,
                 bn_momentum=0.1,
                ):
        super().__init__()
        self.forward_features=forward_features
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, stem_spatial_kernel, stem_spatial_kernel),
                               stride=(conv1_t_stride, stem_spatial_stride, stem_spatial_stride),
                               padding=(conv1_t_size // 2, stem_spatial_kernel // 2, stem_spatial_kernel // 2),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, stride=stage_strides[0])
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=stage_strides[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=stage_strides[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=stage_strides[3])

        self.avgpool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.eps = bn_eps
                m.momentum = bn_momentum
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = out.new_zeros(out.size(0), planes - out.size(1), *out.shape[2:])
        out = torch.cat([out.detach(), zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        if self.forward_features:
            return [x1,x2,x3,x4]
        else:
            x = self.avgpool(x4)

            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x


def _pick_group_norm_groups(num_channels: int, desired_groups: int) -> int:
    num_channels = int(num_channels)
    desired_groups = int(desired_groups)
    if num_channels <= 0:
        raise ValueError(f"num_channels must be > 0, got {num_channels}")
    desired_groups = max(1, min(desired_groups, num_channels))
    for g in range(desired_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def replace_batchnorm_with_groupnorm(module: nn.Module, *, desired_groups: int = 32) -> nn.Module:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm3d)):
            num_channels = int(child.num_features)
            groups = _pick_group_norm_groups(num_channels, desired_groups)
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_channels, affine=True)
            if getattr(child, "affine", False):
                with torch.no_grad():
                    gn.weight.copy_(child.weight)
                    gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            replace_batchnorm_with_groupnorm(child, desired_groups=desired_groups)
    return module


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale, *, norm="batch", group_norm_groups=32, out_channels=1, kernel_size=3, bn_eps=1e-5, bn_momentum=0.1, align_corners=False):
        super().__init__()
        norm = str(norm).lower()
        if norm not in {"batch", "group"}:
            raise ValueError(f"Unknown norm: {norm!r}")

        def _norm2d(num_channels: int) -> nn.Module:
            if norm == "group":
                groups = _pick_group_norm_groups(num_channels, int(group_norm_groups))
                return nn.GroupNorm(num_groups=groups, num_channels=int(num_channels))
            return nn.BatchNorm2d(int(num_channels), eps=bn_eps, momentum=bn_momentum)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], kernel_size, 1, kernel_size // 2, bias=False),
                _norm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], out_channels, 1, 1, 0)
        self.align_corners = align_corners
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear", align_corners=align_corners)

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear", align_corners=self.align_corners)
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


_DEPTHS = {
    10: (BasicBlock, (1, 1, 1, 1)),
    18: (BasicBlock, (2, 2, 2, 2)),
    34: (BasicBlock, (3, 4, 6, 3)),
    50: (Bottleneck, (3, 4, 6, 3)),
    101: (Bottleneck, (3, 4, 23, 3)),
    152: (Bottleneck, (3, 8, 36, 3)),
    200: (Bottleneck, (3, 24, 36, 3)),
}


class VesuviusResNet3D(nn.Module):
    """3D encoder + depth projection + 2D decoder for ink detection.

    All arguments are keyword-only. ``layers`` overrides the depth preset;
    ``block`` may be 'basic' or 'bottleneck' to override its block type.
    ``forward_features=True`` returns the four 3D encoder feature maps.
    ``n_classes`` only sizes the unused legacy classification head (kept for
    checkpoint compatibility); ``out_channels`` sizes the ink prediction head.
    ``stage_strides`` must preserve a spatial factor of two between decoder
    levels; use (depth_stride, 2, 2) to change only depth downsampling.
    Existing hybrid model state_dicts load directly with strict=True.
    """

    def __init__(
        self, *, model_depth=50, layers=None, block=None,
        block_inplanes=(64, 128, 256, 512), widen_factor=1.0,
        n_input_channels=1, n_classes=1039, out_channels=1,
        conv1_t_size=7, conv1_t_stride=1,
        stem_spatial_kernel=7, stem_spatial_stride=2,
        no_max_pool=False, pool_kernel=(1, 3, 3),
        pool_stride=(1, 2, 2), pool_padding=(0, 1, 1),
        stage_strides=(1, 2, 2, 2), shortcut_type='B',
        norm='batch', group_norm_groups=32, bn_eps=1e-5,
        bn_momentum=0.1, with_norm=False, z_projection='max',
        decoder_kernel_size=3, upscale=1, align_corners=False,
        forward_features=False,
    ):
        super().__init__()
        if model_depth not in _DEPTHS:
            raise ValueError(f'model_depth must be one of {tuple(_DEPTHS)}')
        block_cls, preset = _DEPTHS[model_depth]
        if block is not None:
            if block not in {'basic', 'bottleneck'}:
                raise ValueError("block must be 'basic' or 'bottleneck'")
            block_cls = BasicBlock if block == 'basic' else Bottleneck
        layers = tuple(preset if layers is None else layers)
        for name, values in [('layers', layers), ('block_inplanes', block_inplanes)]:
            if len(values) != 4 or any(int(v) != v or v <= 0 for v in values):
                raise ValueError(f'{name} must contain four positive integers')
        if len(stage_strides) != 4:
            raise ValueError('stage_strides must contain four strides')
        for stride in stage_strides[1:]:
            spatial = (stride, stride) if isinstance(stride, int) else tuple(stride)[1:]
            if spatial != (2, 2):
                raise ValueError('Decoder requires spatial stage strides of 2')
        if norm not in {'batch', 'group'}:
            raise ValueError("norm must be 'batch' or 'group'")
        if z_projection not in {'max', 'mean'}:
            raise ValueError("z_projection must be 'max' or 'mean'")
        for kernel in (conv1_t_size, stem_spatial_kernel, decoder_kernel_size):
            if kernel <= 0 or kernel % 2 != 1:
                raise ValueError('Kernel sizes must be positive odd integers')
        if widen_factor <= 0 or any(int(v * widen_factor) < 1 for v in block_inplanes):
            raise ValueError('widen_factor must produce positive channel counts')
        self.with_norm = with_norm
        self.forward_features = forward_features
        self.z_projection = z_projection
        self.backbone = ResNet(
            block_cls, layers, block_inplanes,
            n_input_channels=n_input_channels, n_classes=n_classes,
            conv1_t_size=conv1_t_size, conv1_t_stride=conv1_t_stride,
            no_max_pool=no_max_pool, shortcut_type=shortcut_type,
            widen_factor=widen_factor, forward_features=True,
            stage_strides=stage_strides, stem_spatial_kernel=stem_spatial_kernel,
            stem_spatial_stride=stem_spatial_stride, pool_kernel=pool_kernel,
            pool_stride=pool_stride, pool_padding=pool_padding,
            bn_eps=bn_eps, bn_momentum=bn_momentum,
        )
        if norm == 'group':
            replace_batchnorm_with_groupnorm(self.backbone, desired_groups=group_norm_groups)
        encoder_dims = [int(c * widen_factor) * block_cls.expansion for c in block_inplanes]
        self.decoder = Decoder(
            encoder_dims, upscale, norm=norm, group_norm_groups=group_norm_groups,
            out_channels=out_channels, kernel_size=decoder_kernel_size,
            bn_eps=bn_eps, bn_momentum=bn_momentum, align_corners=align_corners,
        )
        if with_norm:
            self.normalization = (
                nn.GroupNorm(1, n_input_channels) if norm == 'group' else
                nn.BatchNorm3d(n_input_channels, eps=bn_eps, momentum=bn_momentum)
            )

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if x.ndim != 5 or x.shape[1] != self.backbone.conv1.in_channels:
            raise ValueError('Expected (B, C, D, H, W), or (B, D, H, W) for C=1')
        if self.with_norm:
            x = self.normalization(x)
        features = self.backbone(x)
        if self.forward_features:
            return features
        pooled = [torch.max(f, dim=2)[0] if self.z_projection == 'max'
                  else torch.mean(f, dim=2) for f in features]
        return self.decoder(pooled)
