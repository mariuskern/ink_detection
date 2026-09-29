from pathlib import Path

from training.utils import fit

from utils import predict, postprocess_prediction, save_predictions_as_images, save_predictions_as_zarr
from utils.device import get_device
from data import build_dataset, build_dataloader, build_augmentation


def train_noisy_student(cfg, logger):
    output_dir = Path(cfg.path.output_dir)
    tmp_dir = Path(cfg.path.tmp_dir)

    for iteration in range(cfg.train.iterations):
        print(f"Starting Noisy Student iteration {iteration}")

        output_dir_iteration = output_dir / "iteration" / str(iteration)
        tmp_dir_iteration = tmp_dir / "iteration" / str(iteration)

        device = get_device(cfg)

        # Training
        result = fit(
            cfg,
            logger,
            output_dir=output_dir_iteration,
            pseudo_label_version=iteration - 1 if iteration > 0 else None,
            iteration=iteration,
            augmentation=cfg.train.student.aug,
            device=device,
        )

        if "predict" in cfg.train and cfg.train.predict is not None and len(cfg.train.predict) > 0:
            save_predictions(cfg, result.model, output_dir_iteration, tmp_dir_iteration, device)

        generate_pseudo_labels(cfg, result.model, iteration, output_dir_iteration, tmp_dir_iteration, device)


def generate_pseudo_labels(cfg, model, iteration, output_dir, tmp_dir, device):
        # transform = build_augmentation(cfg.train.teacher.aug) if cfg.train.teacher.aug is not None else None
        transforms = []
        if cfg.train.teacher.aug is not None:
            for variant in cfg.train.teacher.aug.variants:
                transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, min_mask_fraction=0.0, transform=transform)
        predict_dataset = build_dataset(cfg, cfg.train.unlabeled, transform=None, split="train" if cfg.train.use_validation_mask else None)
        predict_loader = build_dataloader(cfg, predict_dataset, None)

        for prediction in predict(model, predict_loader, device=device, precision=cfg.train.precision, transforms=transforms, tta_num=cfg.train.teacher.tta_num, cache_dir=tmp_dir):
            # results = predict(model, predict_loader, device=device)
            # # results = predict(model, predict_loader, device=device, tta_transform=transform, tta_num=cfg.train.refinement.tta_num)

            prediction = postprocess_prediction(cfg, prediction)

            save_predictions_as_images(
                output_dir,
                [prediction],
                save_probability=True,
                save_confidence=True,
                save_uncertainty=True,
                save_label=True,
                save_mask=True,
                save_label_not_refined=True,
                save_mask_not_refined=True,
            )

            save_predictions_as_zarr(
                Path(cfg.path.data_dir),
                [prediction],
                iteration,
                save_probability=True,
                save_label=True,
                save_mask=True,
            )


            # results = [prediction]

            # results = postprocess_predictions(
            #     results,
            #     cfg.train.refinement.threshold,
            #     cfg.train.refinement.confidence_threshold,
            #     cfg.train.refinement.uncertainty_threshold,
            #     use_confidence_mask=cfg.train.refinement.use_confidence_mask,
            #     use_uncertainty_mask=cfg.train.refinement.use_uncertainty_mask,
            # )

            # results = postprocess_predictions(
            #     results,
            #     strong_threshold=cfg.train.refinement.strong_threshold,
            #     weak_threshold=cfg.train.refinement.weak_threshold,
            #     confidence_threshold=cfg.train.refinement.confidence_threshold,
            #     uncertainty_threshold=cfg.train.refinement.uncertainty_threshold,
            #     # use_confidence_mask=cfg.train.refinement.use_confidence_mask,
            #     # use_uncertainty_mask=cfg.train.refinement.use_uncertainty_mask,
            #     min_component_size=cfg.train.refinement.min_component_size,
            #     opening_radius=cfg.train.refinement.opening_radius,
            #     closing_radius=cfg.train.refinement.closing_radius,
            #     local_window_sigma=cfg.train.refinement.local_window_sigma,
            #     use_opening=cfg.train.refinement.use_opening,
            #     use_closing=cfg.train.refinement.use_closing,
            #     use_reconstruction=cfg.train.refinement.use_reconstruction,
            #     use_components=cfg.train.refinement.use_components,
            #     use_local_mean=cfg.train.refinement.use_local_mean,
            # )

            # save_predictions_as_images(
            #     output_dir,
            #     results,
            #     save_probability=True,
            #     save_confidence=True,
            #     save_uncertainty=True,
            #     save_label=True,
            #     save_mask=True,
            #     save_label_not_refined=True,
            #     save_mask_not_refined=True,
            # )

            # save_predictions_as_zarr(
            #     Path(cfg.path.data_dir),
            #     results,
            #     iteration,
            #     save_probability=True,
            #     save_label=True,
            #     save_mask=True,
            # )

        # results = predict(model, predict_loader, device=device)
        # # results = predict(model, predict_loader, device=device, tta_transform=transform, tta_num=cfg.train.refinement.tta_num)
        # results = postprocess_predictions(
        #     results,
        #     cfg.train.refinement.threshold,
        #     cfg.train.refinement.confidence_threshold,
        #     cfg.train.refinement.uncertainty_threshold,
        #     use_confidence_mask=cfg.train.refinement.use_confidence_mask,
        #     use_uncertainty_mask=cfg.train.refinement.use_uncertainty_mask,
        # )
        # # results = postprocess_predictions(
        # #     results,
        # #     strong_threshold=cfg.train.refinement.strong_threshold,
        # #     weak_threshold=cfg.train.refinement.weak_threshold,
        # #     confidence_threshold=cfg.train.refinement.confidence_threshold,
        # #     uncertainty_threshold=cfg.train.refinement.uncertainty_threshold,
        # #     # use_confidence_mask=cfg.train.refinement.use_confidence_mask,
        # #     # use_uncertainty_mask=cfg.train.refinement.use_uncertainty_mask,
        # #     min_component_size=cfg.train.refinement.min_component_size,
        # #     opening_radius=cfg.train.refinement.opening_radius,
        # #     closing_radius=cfg.train.refinement.closing_radius,
        # #     local_window_sigma=cfg.train.refinement.local_window_sigma,
        # #     use_opening=cfg.train.refinement.use_opening,
        # #     use_closing=cfg.train.refinement.use_closing,
        # #     use_reconstruction=cfg.train.refinement.use_reconstruction,
        # #     use_components=cfg.train.refinement.use_components,
        # #     use_local_mean=cfg.train.refinement.use_local_mean,
        # # )

        # save_predictions_as_images(
        #     output_dir,
        #     results,
        #     save_probability=True,
        #     save_confidence=True,
        #     save_uncertainty=True,
        #     save_label=True,
        #     save_mask=True,
        #     save_label_not_refined=True,
        #     save_mask_not_refined=True,
        # )

        # save_predictions_as_zarr(
        #     Path(cfg.path.data_dir),
        #     results,
        #     iteration,
        #     save_probability=True,
        #     save_label=True,
        #     save_mask=True,
        # )

        # # prepare_pseudo_tile_manifests(
        # #     cfg.path.data_dir,
        # #     cfg.train.unlabeled,
        # #     version=iteration,
        # #     tile_size=cfg.dataset.tile_size,
        # #     stride=cfg.dataset.stride,
        # #     target=cfg.train.pseudo_target,
        # #     split="train" if cfg.train.use_validation_mask else None
        # # )


def save_predictions(cfg, model, output_dir, tmp_dir, device):
        # transforms = []
        # if cfg.train.teacher.aug is not None:
        #     for variant in cfg.train.teacher.aug.variants:
        #         transforms.append(build_augmentation(variant))
        # predict_dataset = build_dataset(cfg, cfg.train.unlabeled, transform=None, split="train" if cfg.train.use_validation_mask else None)
        predict_dataset = build_dataset(cfg, cfg.train.predict, transform=None, split=None)
        predict_loader = build_dataloader(cfg, predict_dataset, None)

        # for prediction in predict(model, predict_loader, device=device, transforms=transforms, tta_num=cfg.train.refinement.tta_num, cache_dir=output_dir / "tmp"):
        for prediction in predict(model, predict_loader, device=device, precision=cfg.train.precision, transforms=None, tta_num=1, cache_dir=tmp_dir):
            results = [prediction]

            save_predictions_as_images(
                output_dir,
                results,
                save_probability=True,
            )
