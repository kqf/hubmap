import pytest
import numpy as np

from models.metrics import dice, plot


@pytest.fixture
def masks(batch_size, h, w):
    return np.random.randint(0, 2, (batch_size, h, w))


@pytest.fixture
def probas(batch_size, h, w):
    return np.random.uniform(0, 1, (batch_size, h, w))


@pytest.mark.parametrize("batch_size, h, w", [
    (32, 256, 256),
])
@pytest.mark.parametrize("thresholds", [
    0.5,
    np.arange(0.1, 0.9, 0.01)
])
def test_dice(batch_size, masks, probas, thresholds):
    expected_shape = (batch_size,) + np.shape(thresholds)
    assert dice(masks, probas, thresholds).shape == expected_shape


@pytest.mark.parametrize("batch_size, h, w", [
    (32, 256, 256),
])
def test_plots(batch_size, masks, probas):
    thresholds = np.arange(0.1, 0.9, 0.01)
    coefs = dice(masks, probas, thresholds)

    avg, std = coefs.mean(0), coefs.std(0)
    plot(avg, thresholds, std=std)
