import click
import pandas as pd
import tifffile

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
            image = tiff_read(path.with_suffix('.tiff'))
        except FileNotFoundError:
            print(f"Ignoring sample {sample}")
            continue

        # Decode
        mask = rl_decode(encoding, image.shape[:2])

        # Write the file
        tifffile.imwrite(path.with_name(sample + "-mask.png"), mask)


if __name__ == '__main__':
    main()
