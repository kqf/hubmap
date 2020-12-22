import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset


@click.command()
@click.option("--fin", type=cpath(exists=True))
def main(fin):
    train_folders = list(Path(fin).glob("**/*/*/"))
    train = RawDataset(train_folders)

    tile, mask = train[0]
    print(tile.shape, mask.shape)


if __name__ == '__main__':
    main()
