import os

import click
import torch
import random
import numpy as np
from click import Path as cpath

from pathlib import Path
from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model
from environs import Env


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


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
        "trained",
        message.replace(" ", "-")
    )

    model = build_model(
        max_epochs=50,
        logdir=logdir,
        logdir_local=logdir_local
    )
    model.fit(train)
    model.save_params(f_params=os.path.join(logdir_local, 'final.pkl'))


if __name__ == '__main__':
    main()
