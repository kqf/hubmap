from models.dataset import RawDataset
from models.augmentations import transform
from models.model import build_model
from skorch.helper import predefined_split


def test_model(fake_dataset):
    dataset = RawDataset(list(fake_dataset.glob("*/")), transform=transform())

    model = build_model(train_split=predefined_split(dataset))
    model.fit(dataset)

    av, _ = model.thresholds(dataset)
    assert av.shape[0] == len(dataset)
