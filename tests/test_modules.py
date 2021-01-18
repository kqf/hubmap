
import pytest

import torch

from models.model import UNet


@pytest.fixture(scope="module")
def batch(batch_size=128, n_channels=3, imsize=32):
    return torch.rand((batch_size, n_channels, imsize, imsize))


@pytest.mark.parametrize("build_model", [
    UNet,
])
def test_model(build_model, batch):
    batch_size, n_channels, imsize, imsize = batch.shape

    model = build_model()
    assert model(batch).shape == (batch_size, 1, imsize, imsize)
