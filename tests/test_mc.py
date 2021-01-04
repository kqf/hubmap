import matplotlib.pyplot as plt
from models.mc import make_blob


def test_generates():
    plt.imshow(make_blob())
    plt.show()
