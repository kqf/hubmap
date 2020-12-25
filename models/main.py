import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset
from models.augmentations import transform
from models.plot import plot


@click.command()
@click.option("--fin", type=cpath(exists=True))
def main(fin):
    train_folders = list(Path(fin).glob("**/*/*/"))[:5]
    train = RawDataset(train_folders, transform=transform())

    tile, mask = train[0]
    print(tile.shape, mask.shape)
    plot(*zip(*train))


if __name__ == '__main__':
    main()
