from models.dataset import RawDataset


def test_dataset(fake_dataset, size):
    for (tile, mask) in RawDataset(list(fake_dataset.glob("*/"))):
        assert tile.shape == (size, size, 3)
        assert mask.shape == (size, size)
