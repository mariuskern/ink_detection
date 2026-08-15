import torch.nn as nn

from .code_from_omar.shared import TextureStem, DepthCollapse, Head
from .code_from_omar.unetrpp import build_unetrpp_ink


class UNetRpp(nn.Module):
    def __init__(
            self,
            depth,
            tile_size,
            feature_size=16,
            dims=[32, 64, 128, 256],
            depths=[3, 3, 3, 3],
            num_heads=4,
            patch_size=[2, 4, 4],
            pe_mode="2d",
            depth_attn=False,
            da_gamma_init=0.1,
            surface_relative=False,
            per_pixel=False,
            warp_stabilize=False,
            # Stem
            texture_stem=False,
            stem_out_extra=8,
            # Collapse
            collapse="mean",
            # Head
            head_classes=1,
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
            in_ch=in_ch,
            img_size=(D, H, W),
            feature_size=feature_size,
            depths=depths,
            dims=dims,
            num_heads=num_heads,
            patch_size=patch_size,
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