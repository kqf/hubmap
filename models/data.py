import click
import pandas as pd
import tifffile

from multiprocessing import Pool
from pathlib import Path
from click import Path as cpath

from tqdm import tqdm
from PIL import Image
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
        cell_fn = sample / 'full.tiff'
        mask_fn = sample / 'mask.png'

        cell, mask = Image.open(cell_fn).convert('RGB'), Image.open(mask_fn)
        assert cell.size == mask.size
        return cell, mask


def combine_masks(mask_root_dir):
    mask_output = mask_root_dir / 'mask.png'
    if mask_output.exists():
        return

    mask_fn_iter = mask_root_dir.glob('masks/*.png')
    img = Image.open(next(mask_fn_iter))
    for fn in mask_fn_iter:
        mask = Image.open(fn)
        img.paste(mask, (0, 0), mask)

    img.save(mask_output)


def tiff_read(filename):
    image = tifffile.imread(filename)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    return image


@click.command()
@click.option("--codes", type=cpath(exists=True), default="data/train.csv")
@click.option("--opath", type=cpath(exists=True), default="data/train")
def main(codes, opath):
    # Combine masks into one
    df = pd.read_csv(codes)
    print(df.head())
    for _, (sample, encoding) in tqdm(df.iterrows(), total=len(df)):
        path = Path(opath) / sample
        try:
            image = tiff_read(path.with_suffix('.tiff'))
        except FileNotFoundError:
            print(f"Ignoring sample {sample}")
            continue
        mask = rl_decode(encoding, image.shape[:2])
        Image.fromarray(mask).save(path.with_name(sample + "-mask.png"))


if __name__ == '__main__':
    main()
