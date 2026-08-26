from .label_refiner import LabelRefiner


def postprocess_predictions(
    predictions,
    strong_threshold,
    weak_threshold,
    confidence_threshold,
    uncertainty_threshold,
    # use_confidence_mask,
    # use_uncertainty_mask,
    min_component_size,
    opening_radius,
    closing_radius,
    local_window_sigma,
    use_opening,
    use_closing,
    use_reconstruction,
    use_components,
    use_local_mean,
):
    results = []

    refiner = LabelRefiner(
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        confidence_threshold=confidence_threshold,
        uncertainty_threshold=uncertainty_threshold,
        # use_confidence_mask=use_confidence_mask,
        # use_uncertainty_mask=use_uncertainty_mask,
        min_component_size=min_component_size,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        local_window_sigma=local_window_sigma,
        use_opening=use_opening,
        use_closing=use_closing,
        use_reconstruction=use_reconstruction,
        use_components=use_components,
        use_local_mean=use_local_mean,
    )

    for prediction in predictions:
        label_not_refined, mask_not_refined, refined_label, refined_mask, debug = refiner.refine(prediction)

        prediction.pseudo_label = refined_label
        prediction.pseudo_mask = refined_mask

        prediction.pseudo_label_not_refined = label_not_refined
        prediction.pseudo_mask_not_refined = mask_not_refined

        prediction.debug = debug

        results.append(prediction)

    return results



# def postprocess_predictions(
#     predictions,
#     threshold,
#     confidence_threshold,
#     uncertainty_threshold,
#     use_confidence_mask=True,
#     use_uncertainty_mask=True,
# ):
#     results = []

#     for prediction in predictions:
#         label = (prediction.probability >= threshold).astype(np.uint8)

#         mask = np.ones_like(label, dtype=bool)

#         if use_confidence_mask:
#             mask &= prediction.confidence > confidence_threshold

#         if use_uncertainty_mask:
#             mask &= prediction.uncertainty < uncertainty_threshold

#         prediction.pseudo_label = label
#         prediction.pseudo_mask = mask

#         results.append(prediction)

#     return results
