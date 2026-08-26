from models.unetrpp import UNetRpp
from models.timesformer import TimeSformer


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
                cfg.model.num_classes,
                cfg.model.channels,
                cfg.model.depth,
                cfg.model.heads,
                cfg.model.dim_head,
                cfg.model.attn_dropout,
                cfg.model.ff_dropout,
                cfg.model.with_norm
            )