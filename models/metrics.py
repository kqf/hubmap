import numpy as np
import matplotlib.pyplot as plt


def iou_approx(true_masks, probas, padding=0):
    _, sx, sy = probas.shape
    true_masks = true_masks[:, padding:sx - padding, padding:sy - padding]
    probas = probas[:, padding:sx - padding, padding:sy - padding]
    preds = 1 / (1 + np.exp(-probas))

    approx_intersect = np.sum(np.minimum(preds, true_masks), axis=(1, 2))
    approx_union = np.sum(np.maximum(preds, true_masks), axis=(1, 2))
    return np.mean(approx_intersect / approx_union)


def dice(probas, masks, th=np.arange(0.1, 0.9, 0.01)):
    # Ensure threshold dimension: [b, h, w] -> [b, h, w, 1]
    probas = probas[..., None]
    masks = masks[..., None]

    # [b, h, w, t]
    preds = probas > th

    # [b, h, w, t]
    inter = (preds * masks).sum(axis=(1, 2))

    # [b, h, w, t]
    union = (preds + masks).sum(axis=(1, 2))

    # Handle the situation when union is zero
    regularized_inter = np.where(union > 0.0, inter, np.zeros_like(union))
    regularized_union = np.where(union > 0.0, union, np.ones_like(union))

    # Remove the extra dimension when th is scalar
    return np.squeeze(2 * regularized_inter / regularized_union)


def plot(mean, thresholds, std=None, cfg=None, fname='thresholds.png'):
    cfg = cfg or dict(
        ls='',
        marker='o',
        markersize=2,
        capsize=1,
        capthick=1,
    )
    plt.errorbar(thresholds, mean, yerr=std, **cfg)
    plt.ylabel('dice coefficient')
    plt.xlabel('thresholds')

    peak = thresholds[mean.argmax()]
    plt.axvline(x=peak, ymax=mean.max(), color='k', ls='--')
    plt.text(peak, mean.mean(), f'threshold = {peak:.4g}')

    plt.show()
    plt.savefig(fname)
