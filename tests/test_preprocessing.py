import pytest
import numpy as np


from models.preprocess import pad


@pytest.fixture
def image(size, channels):
    shape = (size, size)
    if channels is not None:
        shape += (channels,)
    return np.random.randint(0, 255, shape)


@pytest.mark.parametrize("size", [
    256,
    1024,
    1023,
])
@pytest.mark.parametrize("channels", [
    None,
    3
])
def test_imsize(image, reduction=1):
    padded = pad(image, reduction=1)

    # Check if first two dimensions are padded now
    assert padded.shape[0] >= image.shape[0]
    assert padded.shape[1] >= image.shape[1]

    # Check if the remaning dimeensions agree
    for p, o in zip(padded.shape[2:], image.shape[2:]):
        assert p == o
