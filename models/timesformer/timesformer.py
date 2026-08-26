import torch
from torch import nn
from timesformer_pytorch import TimeSformer as _TimeSformer


class TimeSformer(nn.Module):
    def __init__(
        self,
        dim=512,
        image_size=64,
        patch_size=16,
        num_frames=26,
        num_classes=16,
        channels=1,
        depth=8,
        heads=6,
        dim_head=64,
        attn_dropout=0.1,
        ff_dropout=0.1,
        with_norm=False
    ):
        super().__init__()
        
        self.backbone = _TimeSformer(
            dim=dim,
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_classes,
            channels=channels,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        self.with_norm = with_norm

        if self.with_norm:
            self.norm = nn.BatchNorm3d(num_features=1)

    def encode(self, x):
        raise NotImplementedError()

    def decode(self, features):
        raise NotImplementedError()

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]

        if self.with_norm:
            x = self.norm(x)
        
        x = torch.permute(x, (0, 2, 1, 3, 4))
        x = self.backbone(x)
        x = x.view(-1, 1, 4, 4)

        return x