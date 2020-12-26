import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def tensor2img(t, padding=0):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    # return t * std + mu if t.shape[0] > 1 else t
    img = to_pil_image(t * std + mu if t.shape[0] > 1 else t)
    w, h = img.size
    return np.array(img.crop((padding, padding, w - padding, h - padding)))


def plot(*imgs):
    fig, axes = plt.subplots(len(imgs), len(imgs[0]), figsize=(12, 5))

    # If there is a single row in the data
    if len(imgs) == 1:
        axes = [axes]

    for row, raxes in zip(imgs, axes):
        for i, (image, ax) in enumerate(zip(row, raxes)):
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Sample {i}")
            try:
                ax.imshow(image)
            except TypeError:
                ax.imshow(tensor2img(image))
    plt.show()
    return axes


def batches(dataset, batch_size):
    batch = []
    for pair in dataset:
        if len(batch) < batch_size:
            compare([pair[0]], [pair[1]])
            batch.append(pair)
            continue
        yield list(zip(*batch))
        batch = []


def compare(image, mask):
    plt.imshow(tensor2img(image))
    plt.imshow(tensor2img(mask), alpha=0.6)
    plt.show()


def glance(dataset, batch_size, pfunc=plot):
    for batch in batches(dataset, batch_size):
        pfunc(*batch)
