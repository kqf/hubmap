import click
import cv2

from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class RawDataset(Dataset):
    """Constructs dataset"""

    def __init__(self, sample_dirs):
        super().__init__()
        self.sample_dirs = sample_dirs

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        cell_fn = (sample_dir / 'images' / sample_dir.name).with_suffix('.png')
        mask_fn = sample_dir / 'mask.png'

        cell, mask = Image.open(cell_fn).convert('RGB'), Image.open(mask_fn)
        assert cell.size == mask.size
        return cell, mask


class Dataset(Dataset):
    def __init__(self, sample_dirs, transform=None):
        super().__init__()
        self.sample_dirs = sample_dirs
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        cell_fn = (sample_dir / 'images' / sample_dir.name).with_suffix('.png')
        mask_fn = sample_dir / 'mask.png'

        # Read an image with OpenCV
        image = cv2.imread(str(cell_fn))
        mask = cv2.imread(str(mask_fn), cv2.IMREAD_GRAYSCALE)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented["mask"]
        return image, (mask == 255).float()


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


@click.command()
@click.option("--path", type=click.Path(exists=True), default="data/train")
def main(path):
    # Combine masks into one
    samples_dirs = list(d for d in Path(path).iterdir() if d.is_dir())
    with Pool() as pool, tqdm(total=len(samples_dirs)) as pbar:
        for _ in tqdm(pool.imap_unordered(combine_masks, samples_dirs)):
            pbar.update()


if __name__ == '__main__':
    main()
