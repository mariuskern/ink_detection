from typing import Sequence, Tuple, Union

import torch.nn as nn

from .code_from_omar.shared import TextureStem, DepthCollapse, Head
from .code_from_omar.unetrpp import build_unetrpp_ink


class UNetRpp(nn.Module):
    def __init__(
            self,
            depth: int,
            tile_size: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            norm_name: Union[Tuple, str] = "instance",
            depths: Sequence[int] = (3, 3, 3, 3),
            dims: Sequence[int] = (32, 64, 128, 256),
            patch_size: Sequence[int] = (2, 4, 4),
            proj_size: Sequence[int] = (64, 64, 64, 32),
            encoder_dropout_rate=0.0,
            attention_dropout_rate=0.15,
            conv_dropout_rate=0.0,
            conv_projection_dropout_rate=0.1,
            drop_path_rate=0.0,
            pe_mode: str = "2d",
            depth_attn: bool = False,
            da_gamma_init: float = 0.1,
            surface_relative: bool = False,
            per_pixel: bool = False,
            warp_stabilize: bool = False,
            # Stem
            texture_stem=False,
            stem_out_extra=8,
            # Collapse
            collapse="mean",
            # Head
            head_classes=1,
            decoder_depths: Sequence[int] = (3, 3, 3),
    ):
        super().__init__()
        
        if texture_stem:
            self.stem = TextureStem(out_extra=stem_out_extra)
            in_ch = self.stem.out_ch
        else:
            self.stem = None
            in_ch = 1

        D, H, W = depth, tile_size, tile_size
        
        self.backbone = build_unetrpp_ink(
            in_channels=in_ch,
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            norm_name=norm_name,
            depths=depths,
            decoder_depths=decoder_depths,
            dims=dims,
            img_size=(D, H, W),
            patch_size=patch_size,
            proj_size=proj_size,
            encoder_dropout_rate=encoder_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            conv_dropout_rate=conv_dropout_rate,
            conv_projection_dropout_rate=conv_projection_dropout_rate,
            drop_path_rate=drop_path_rate,
            pe_mode=pe_mode,
            depth_attn=depth_attn,
            da_gamma_init=da_gamma_init,
            surface_relative=surface_relative,
            per_pixel=per_pixel,
            warp_stabilize=warp_stabilize,
        )

        feat_ch = self.backbone.out_channels

        self.collapse = DepthCollapse(
            depth=D,
            mode=collapse,
            channels=feat_ch,
            per_pixel=per_pixel,
            warp_stabilize=warp_stabilize
        )

        self.head = Head(feat_ch, out_ch=head_classes)

    # def forward(self, x):
    #     if self.stem is not None:
    #         x = self.stem(x)

    #     feat = self.backbone(x) # [B,C,D,H,W]
    #     s_hat = getattr(self.backbone, "last_s_hat", None)
    #     x2d = self.collapse(feat, s_hat) # [B,C,H,W]
    #     out = self.head(x2d) # [B,1,H,W]
    #     return out

    def encode(self, x):
        if self.stem is not None:
            x = self.stem(x)

        # Encoder des UNETR++
        return self.backbone.encode(x)

    def decode(self, features):
        # UNETR++ Decoder
        feat = self.backbone.decode(*features)  # [B, C, D, H, W]

        # 3D -> 2D
        s_hat = getattr(self.backbone, "last_s_hat", None)
        x2d = self.collapse(feat, s_hat)        # [B, C, H, W]

        # 2D segmentation head
        out = self.head(x2d)                    # [B, 1, H, W]

        return out

    def forward(self, x):
        features = self.encode(x)
        return self.decode(features)
