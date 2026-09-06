from models.unetrpp import UNetRpp
from models.timesformer import TimeSformer
from models.vesuvius_models import VesuviusUNet
from models.vesuvius_models import VesuviusResNet3D


def build_model(cfg):
    match cfg.model.name:
        case "unetrpp":
            return UNetRpp(
                cfg.dataset.depth,
                cfg.dataset.tile_size,
                cfg.model.feature_size,
                cfg.model.dims,
                cfg.model.depths,
                cfg.model.num_heads,
                cfg.model.patch_size,
                cfg.model.pe_mode,
                cfg.model.depth_attn,
                cfg.model.da_gamma_init,
                cfg.model.surface_relative,
                cfg.model.per_pixel,
                cfg.model.warp_stabilize,
                # Stem
                cfg.model.texture_stem,
                cfg.model.stem_out_extra,
                # Collapse
                cfg.model.collapse,
                # Head
                cfg.model.head_classes,
            )
        case "timesformer":
            return TimeSformer(
                cfg.model.dim,
                cfg.dataset.tile_size, # cfg.model.image_size,
                cfg.model.patch_size,
                cfg.dataset.depth, # cfg.model.num_frames,
                # cfg.model.num_classes,
                cfg.model.channels,
                cfg.model.depth,
                cfg.model.heads,
                cfg.model.dim_head,
                cfg.model.attn_dropout,
                cfg.model.ff_dropout,
                cfg.model.with_norm
            )
        case "vesuvius_unet":
            return VesuviusUNet(
                in_channels=cfg.model.in_channels,
                out_channels=cfg.model.out_channels,
                z_projection=cfg.model.z_projection,
            )
        case "vesuvius_resnet3d":
            return VesuviusResNet3D(
                model_depth=cfg.model.model_depth,
                n_input_channels=cfg.model.n_input_channels,
                forward_features=cfg.model.forward_features,
                n_classes=cfg.model.n_classes,
                with_norm=cfg.model.with_norm,
                norm=cfg.model.norm,
                group_norm_groups=cfg.model.group_norm_groups,
                upscale=cfg.model.upscale
            )