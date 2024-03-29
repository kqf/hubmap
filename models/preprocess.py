import cv2
import click
import tifffile
import numpy as np
import pandas as pd

from pathlib import Path
from click import Path as cpath

from tqdm import tqdm

from models.encoding import decode


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


def is_saturated(img, s_th=40, sz=256):
    p_th = 200 * sz // 256
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (s > s_th).sum() <= p_th or img.sum() <= p_th


def write(img, tilepath):
    # _, png = cv2.imencode(".png", img)
    tilepath.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tilepath), img)


def dump_averages(x, x2, fout):
    # image stats
    mean = np.stack(x).mean(0)
    std = np.sqrt(np.stack(x2).mean(0) - mean**2)

    # dump the stats
    df = pd.DataFrame({"mean": mean, "std": std})

    print(df.T)
    df.to_csv(Path(fout).with_suffix(".csv"), index=False)


@click.command()
@click.option("--codes", type=cpath(exists=True), default="data/train.csv")
@click.option("--fin", type=cpath(exists=True))
@click.option("--fout", type=cpath(exists=False))
def main(codes, fin, fout):
    # Combine masks into one
    df = pd.read_csv(codes)
    print(df.head())

    x, x2 = [], []
    for _, (sample, encoding) in tqdm(df.iterrows(), total=len(df)):
        path = Path(fin) / sample

        # Read if possible
        try:
            # Read -> [h, w, c]
            image = tiff_read(path.with_suffix(".tiff"))
        except FileNotFoundError:
            print(f"Ignoring sample {sample}")
            continue

        # Decode -> [h, w]
        shape = image.shape[:2]
        mask = decode(encoding, shape[::-1])

        samples = tile(image, interp=cv2.INTER_AREA)
        masks = tile(mask, interp=cv2.INTER_NEAREST)

        out_folder = Path(fout) / sample
        for i, (tsample, tmask) in enumerate(zip(samples, masks)):
            if is_saturated(tsample):
                continue

            # Normalization
            img = tsample / 255.0
            x.append(img.reshape(-1, 3).mean(0))
            x2.append((img ** 2).reshape(-1, 3).mean(0))

            bgr = cv2.cvtColor(tsample, cv2.COLOR_RGB2BGR)
            write(bgr, tilepath=out_folder / f"{i}" / "tile.png")
            write(tmask, tilepath=out_folder / f"{i}" / "mask.png")

    dump_averages(x, x2, fout)


if __name__ == "__main__":
    main()
