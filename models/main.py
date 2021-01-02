import os
import click
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model
from environs import Env


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--message", type=str)
def main(fin, message):
    if not message:
        raise IOError("Please provide the train message")

    train_folders = list(Path(fin).glob("*/*"))
    train = RawDataset(train_folders, transform=transform(train=False))

    env = Env()
    env.read_env()
    logdir = os.path.join(
        env("TENSORBOARD_DIR", "."),
        message.replace(" ", "-")
    )
    logdir_local = os.path.join(
        env("TENSORBOARD_DIR", "."),
        message.replace(" ", "-")
    )

    model = build_model(
        max_epochs=1,
        logdir=logdir,
        logdir_local=logdir_local
    )
    model.fit(train)
    model.save_params(f_params=os.path.join(logdir_local, 'final.pkl'))


if __name__ == '__main__':
    main()
