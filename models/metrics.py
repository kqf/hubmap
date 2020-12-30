import numpy as np


def iou_approx(true_masks, probas, padding=0):
    _, sx, sy = probas.shape
    true_masks = true_masks[:, padding:sx - padding, padding:sy - padding]
    probas = probas[:, padding:sx - padding, padding:sy - padding]
    preds = 1 / (1 + np.exp(-probas))

    approx_intersect = np.sum(np.minimum(preds, true_masks), axis=(1, 2))
    approx_union = np.sum(np.maximum(preds, true_masks), axis=(1, 2))
    return np.mean(approx_intersect / approx_union)
