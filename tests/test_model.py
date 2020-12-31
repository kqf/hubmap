from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model


def test_model(fake_dataset):
    dataset = RawDataset(list(fake_dataset.glob("*/")), transform=transform())

    model = build_model()
    model.fit(dataset)
