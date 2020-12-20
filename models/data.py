import cv2
import click
import tifffile
import numpy as np
import pandas as pd

from pathlib import Path
from click import Path as cpath

from tqdm import tqdm
from torch.utils.data import Dataset

from models.encoding import rl_decode


class RawDataset(Dataset):
    """Constructs dataset"""

    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_fn = sample
        mask_fn = sample.with_name(sample.stem + '-mask.png')
        return tiff_read(sample_fn), tiff_read(mask_fn)


def tiff_read(filename):
    image = tifffile.imread(filename)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    return image


def pad(img, sz=256, reduction=4):
    # add padding to make the image dividable into tiles
    w, h = img.shape[:2]

    padw = (reduction * sz - w % (reduction * sz)) % (reduction * sz)
    padh = (reduction * sz - h % (reduction * sz)) % (reduction * sz)

    padding = [[padw // 2, padw - padw // 2], [padh // 2, padh - padh // 2]]

    # Add zero padding for the remaining dimensions
    for _ in img.shape[2:]:
        padding.append([0, 0])

    return np.pad(img, padding, constant_values=0)


def resize(img, sz=256, reduction=4, interp=cv2.INTER_AREA):
    reduced_shape = (img.shape[1] // reduction, img.shape[0] // reduction)
    return cv2.resize(img, reduced_shape, interpolation=interp)


def tile(img, sz=256, reduction=4, interp=cv2.INTER_AREA):
    img = pad(img, sz=sz, reduction=4)
    img = resize(img, sz=sz, reduction=reduction, interp=interp)

    nch = 3 if len(img.shape) == 3 else 1

    # [h, w, c] -> [h / sz, sz, w / sz, sz, c)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, nch)

    # [h / sz, sz, w / sz, sz, c).T -> [h / sz, w / sz, sz, sz, c)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, nch)
    return np.squeeze(img)


@click.command()
@click.option("--codes", type=cpath(exists=True), default="data/train.csv")
@click.option("--opath", type=cpath(exists=True), default="data/train")
def main(codes, opath):
    # Combine masks into one
    df = pd.read_csv(codes)
    print(df.head())
    for _, (sample, encoding) in tqdm(df.iterrows(), total=len(df)):
        path = Path(opath) / sample

        # Read if possible
        try:
            # Read -> [h, w, c]
            image = tiff_read(path.with_suffix('.tiff'))
        except FileNotFoundError:
            print(f"Ignoring sample {sample}")
            continue

        # Decode -> [h, w]
        mask = rl_decode(encoding, image.shape[:2]).T

        samples = tile(image, interp=cv2.INTER_AREA)
        masks = tile(mask, interp=cv2.INTER_NEAREST)

        for i, (tsample, tmask) in enumerate(zip(samples, masks)):
            pass

        # Write the file
        tifffile.imwrite(path.with_name(sample + "-mask.png"), mask)


if __name__ == '__main__':
    main()
