import cv2
import numpy as np


# def create_training_mask(label, margin=50):
#     label = (label > 0).astype(np.uint8) * 255

#     kernel_size = 2 * margin + 1
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

#     training_mask = cv2.dilate(label, kernel)

#     return training_mask


def create_training_mask(label, margin_factor=0.2):
    if label.ndim == 3:
        label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
    
    label = (label > 0).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(label, connectivity=8)

    training_mask = np.zeros_like(label)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        glyph_size = max(w, h)

        margin = max(1, int(glyph_size * margin_factor))

        kernel_size = 2 * margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        glyph_mask = (labels == i).astype(np.uint8) * 255
        glyph_mask = cv2.dilate(glyph_mask, kernel)

        training_mask = cv2.bitwise_or(training_mask, glyph_mask)

    return training_mask
