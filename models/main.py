import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset
from models.augmentations import transform
from models.plot import glance, compare


@click.command()
@click.option("--fin", type=cpath(exists=True))
def main(fin):
    train_folders = list(Path(fin).glob("*/*"))[:20]
    train = RawDataset(train_folders, transform=transform(train=False))

    tile, mask = train[0]
    print(tile.shape, mask.shape)

    for image, mask in train:
        compare(image, mask)


if __name__ == '__main__':
    main()
