import numpy as np


def make_blob(h=256, w=256, lam=5, area=8, std_area=1):
    nblobs = np.random.poisson(lam)
    mask = np.zeros((h, w))

    Y, X = np.ogrid[:h, :w]

    cx = np.random.randint(0, h, nblobs).reshape(1, 1, nblobs)
    cy = np.random.randint(0, w, nblobs).reshape(1, 1, nblobs)
    radii = np.random.normal(area, std_area, nblobs)

    dists = np.sqrt((X[..., None] - cx) ** 2 + (Y[..., None] - cy) ** 2)

    mask = dists <= radii
    return mask.sum(axis=-1).astype(bool)


def blob2image(blob, channels=3, epsilon=0.1):
    h, w = blob.shape

    extended = blob[..., None]

    # Add a small term to add noise to the empty regions
    noise = np.random.poisson(extended + epsilon, size=(h, w, channels))

    # Convet to image scale
    return extended + noise * 255
