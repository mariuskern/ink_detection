from models.unetrpp import UNetRpp
from models.timesformer import TimeSformer
from models.vesuvius_models import VesuviusUNet
from models.vesuvius_models import VesuviusResNet3D


def build_model(cfg):
    match cfg.model.name:
        case "unetrpp":
            return UNetRpp(
                depth=cfg.dataset.depth,
                tile_size=cfg.dataset.tile_size,
                feature_size=cfg.model.feature_size,
                hidden_size=cfg.model.hidden_size,
                num_heads=cfg.model.num_heads,
                norm_name=cfg.model.norm_name,
                depths=cfg.model.depths,
                dims=cfg.model.dims,
                patch_size=cfg.model.patch_size,
                proj_size=cfg.model.proj_size,
                encoder_dropout_rate=cfg.model.encoder_dropout_rate,
                attention_dropout_rate=cfg.model.attention_dropout_rate,
                conv_dropout_rate=cfg.model.conv_dropout_rate,
                conv_projection_dropout_rate=cfg.model.conv_projection_dropout_rate,
                drop_path_rate=cfg.model.drop_path_rate,
                pe_mode=cfg.model.pe_mode,
                depth_attn=cfg.model.depth_attn,
                da_gamma_init=cfg.model.da_gamma_init,
                surface_relative=cfg.model.surface_relative,
                per_pixel=cfg.model.per_pixel,
                warp_stabilize=cfg.model.warp_stabilize,
                # Stem
                texture_stem=cfg.model.texture_stem,
                stem_out_extra=cfg.model.stem_out_extra,
                # Collapse
                collapse=cfg.model.collapse,
                # Head
                head_classes=cfg.model.head_classes,
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