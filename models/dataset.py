import cv2
from torch.utils.data import Dataset


def channel_first(**kwargs):
    output = {}
    for k, v in kwargs.items():
        output[k] = v.transpose(2, 0, 1) if v.ndim == 3 else v
    return output


class RawDataset(Dataset):
    def __init__(self, samples, transform=channel_first):
        super().__init__()
        self.samples = samples
        self.transform = transform or channel_first

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample / "tile.png"))
        mask = cv2.imread(str(sample / "mask.png"), cv2.IMREAD_GRAYSCALE)

        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented["mask"]

        return image, mask
