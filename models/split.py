import json
import click

from operator import itemgetter
from pathlib import Path
from click import Path as cpath
from sklearn.model_selection import KFold

SEED = 137


@click.command()
@click.option("--fin", type=cpath(exists=True))
@click.option("--fout", type=cpath(exists=False))
def main(fin, fout):
    folders = [str(p) for p in Path(fin).glob("*/*")]
    cv = KFold(n_splits=5, random_state=SEED, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(cv.split(folders)):
        train, test = itemgetter(*train_idx), itemgetter(*test_idx)
        with open(Path(fout) / f"fold{i}.json", 'w') as f:
            json.dump({"train": train(folders), "test": test(folders)}, f)


if __name__ == '__main__':
    main()
