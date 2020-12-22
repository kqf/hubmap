import cv2
from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, samples):
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample / "tile.png"))
        mask = cv2.imread(str(sample / "mask.png"), cv2.IMREAD_GRAYSCALE)
        return image, mask
