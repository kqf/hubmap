import pytest
from models.dataset import RawDataset, channel_first
from models.augmentations import transform


@pytest.mark.parametrize("transform", [
    channel_first,
    transform(train=True),
    transform(train=False),
])
def test_dataset(fake_dataset, size, transform):
    dataset = RawDataset(list(fake_dataset.glob("*/")), transform=transform)

    for (tile, mask) in dataset:
        assert tile.shape == (3, size, size)
        assert mask.shape == (size, size)
