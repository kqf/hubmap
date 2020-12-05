import numpy as np


def rl_decode(code, shape):
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
