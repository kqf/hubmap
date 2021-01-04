import pytest
import tempfile

from pathlib import Path

from models.preprocess import write
from models.mc import make_blob, blob2image


@pytest.fixture
def size():
    return 256


@pytest.fixture
def fake_dataset(size=256, nfiles=5):
    with tempfile.TemporaryDirectory() as dirname:
        path = Path(dirname)
        for i in range(nfiles):
            mask = make_blob(size, size)
            write(mask, path / str(i) / "mask.png")

            tile = blob2image(mask)
            write(tile, path / str(i) / "tile.png")

        yield path
