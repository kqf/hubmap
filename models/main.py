import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset


@click.command()
@click.option("--trainp", type=cpath(exists=True), default="data/train")
@click.option("--testp", type=cpath(exists=True), default="data/test")
def main(trainp, testp):
    train_folders = list(Path(trainp).glob("**/*/*/"))
    train = RawDataset(train_folders)

    tile, mask = train[0]
    print(tile.shape, mask.shape)


if __name__ == '__main__':
    main()
