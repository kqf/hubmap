import click
import torch
import random
import numpy as np

from pathlib import Path
from click import Path as cpath
from sklearn.model_selection import train_test_split
from skorch.helper import predefined_split
from functools import partial

from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model
from models.metrics import dice, plot


SEED = 137

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--logdir", type=str)
def main(fin, logdir):
    folders = list(Path(fin).glob("*/*"))
    ftrain, ftest = train_test_split(folders, random_state=SEED)
    train = RawDataset(ftrain, transform=transform(train=True))
    test = RawDataset(ftest, transform=transform(train=False))

    model = build_model(
        max_epochs=60,
        logdir=logdir,
        train_split=predefined_split(test),
    )
    model.fit(train)

    th = np.arange(0.1, 0.9, 0.01)
    mean, std = model.thresholds(test, partial(dice, th=th))
    plot(mean, thresholds=th)


if __name__ == '__main__':
    main()
