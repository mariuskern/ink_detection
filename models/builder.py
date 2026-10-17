from hydra.utils import get_class

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
                decoder_depths=cfg.model.decoder_depths,
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
                features_per_stage=cfg.model.features_per_stage,
                n_blocks_per_stage=cfg.model.n_blocks_per_stage,
                strides=cfg.model.strides,
                kernel_sizes=cfg.model.kernel_sizes,
                n_conv_per_stage_decoder=cfg.model.n_conv_per_stage_decoder,
                conv_bias=cfg.model.conv_bias,
                norm_op=get_class(cfg.model.norm_op) if cfg.model.norm_op is not None else None,
                norm_op_kwargs=cfg.model.norm_op_kwargs,
                dropout_op=get_class(cfg.model.dropout_op) if cfg.model.dropout_op is not None else None,
                dropout_op_kwargs=cfg.model.dropout_op_kwargs,
                nonlin=get_class(cfg.model.nonlin) if cfg.model.nonlin is not None else None,
                nonlin_kwargs=cfg.model.nonlin_kwargs,
                basic_encoder_block=cfg.model.basic_encoder_block,
                bottleneck_block=cfg.model.bottleneck_block,
                basic_decoder_block=cfg.model.basic_decoder_block,
                do_stem=cfg.model.do_stem,
                stem_channels=cfg.model.stem_channels,
                bottleneck_channels=cfg.model.bottleneck_channels,
                stochastic_depth_p=cfg.model.stochastic_depth_p,
                squeeze_excitation=cfg.model.squeeze_excitation,
                squeeze_excitation_reduction_ratio=cfg.model.squeeze_excitation_reduction_ratio,
                squeeze_excitation_type=cfg.model.squeeze_excitation_type,
                squeeze_excitation_add_maxpool=cfg.model.squeeze_excitation_add_maxpool,
                upsample_mode=cfg.model.upsample_mode,
                target_name=cfg.model.target_name,
                head_bias=cfg.model.head_bias,
            )
        case "vesuvius_resnet3d":
            return VesuviusResNet3D(
                model_depth=cfg.model.model_depth,
                layers=cfg.model.layers,
                block=cfg.model.block,
                block_inplanes=cfg.model.block_inplanes,
                widen_factor=cfg.model.widen_factor,
                n_input_channels=cfg.model.n_input_channels,
                n_classes=cfg.model.n_classes,
                out_channels=cfg.model.out_channels,
                conv1_t_size=cfg.model.conv1_t_size,
                conv1_t_stride=cfg.model.conv1_t_stride,
                stem_spatial_kernel=cfg.model.stem_spatial_kernel,
                stem_spatial_stride=cfg.model.stem_spatial_stride,
                no_max_pool=cfg.model.no_max_pool,
                pool_kernel=cfg.model.pool_kernel,
                pool_stride=cfg.model.pool_stride,
                pool_padding=cfg.model.pool_padding,
                stage_strides=cfg.model.stage_strides,
                shortcut_type=cfg.model.shortcut_type,
                norm=cfg.model.norm,
                group_norm_groups=cfg.model.group_norm_groups,
                bn_eps=cfg.model.bn_eps,
                bn_momentum=cfg.model.bn_momentum,
                with_norm=cfg.model.with_norm,
                z_projection=cfg.model.z_projection,
                decoder_kernel_size=cfg.model.decoder_kernel_size,
                upscale=cfg.model.upscale,
                align_corners=cfg.model.align_corners,
                forward_features=cfg.model.forward_features,
                encoder_dropout_rate=cfg.model.encoder_dropout_rate,
                decoder_dropout_rate=cfg.model.decoder_dropout_rate,
                drop_path_rate=cfg.model.drop_path_rate,
            )
