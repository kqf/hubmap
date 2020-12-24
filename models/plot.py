import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def tensor2img(t, padding=16):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    img = to_pil_image(t * std + mu if t.shape[0] > 1 else t)
    w, h = img.size
    return img.crop((padding, padding, w - padding, h - padding))


def plot(*cells):
    fig, axes = plt.subplots(len(cells), len(cells[0]), figsize=(12, 5))

    # If there is a single row in the data
    if len(cells) == 1:
        axes = [axes]

    for row, raxes in zip(cells, axes):
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
