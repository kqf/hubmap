import numpy as np


def decode(code, shape):
    """
    Decode the run-lenght encoded string into an image of the given shape.
    Parameters
    ----------
    code: str
        run-length as string formated (start length)
    shape: tuple
        (width,height) of the output array
    Returns
    -------
        An array, of shape ``shape`` where 0 corresponds to background
        and 1 to mask.
    References
    ----------
    `The original implementation on kaggle
        <https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode>`_
    Examples
    --------
    >>>
    """
    s = code.split()
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo: hi] = 1

    return img.reshape(shape).T


def encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    This simplified method requires first and last pixel to be zero
    source: https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
    """
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
