import click
from click import Path as cpath

from pathlib import Path
from data import RawDataset


@click.command()
@click.option("--trainp", type=cpath(exists=True), default="data/train")
@click.option("--testp", type=cpath(exists=True), default="data/test")
def main(trainp, testp):
    train = RawDataset(list(Path(trainp).glob("*.tiff")))
    print(train)
    for x in train:
        print(x)


if __name__ == '__main__':
    main()
