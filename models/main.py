import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model


@click.command()
@click.option("--fin", type=cpath(exists=True))
def main(fin):
    train_folders = list(Path(fin).glob("*/*"))
    train = RawDataset(train_folders, transform=transform(train=False))

    model = build_model()
    model.fit(train)
    model.save_params(f_params='final.pkl')


if __name__ == '__main__':
    main()
