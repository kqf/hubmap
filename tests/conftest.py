import pytest
import tempfile
import numpy as np

from pathlib import Path

from models.preprocess import write
# directory and contents have been removed


@pytest.fixture
def size():
    return 256


@pytest.fixture
def fake_dataset(size=256, nfiles=5):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        for i in range(nfiles):
            tile = np.random.randint(0, 255, (size, size, 3))
            write(tile, path / str(i) / "tile.png")

            mask = np.random.randint(0, 255, (size, size))
            write(mask, path / str(i) / "mask.png")
        yield path
