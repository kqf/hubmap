import matplotlib.pyplot as plt
from models.mc import make_blob, blob2image


def test_generates():
    blob = make_blob()
    plt.imshow(blob)

    img = blob2image(blob)
    plt.imshow(img)
