import cv2
import torch
import numpy as np
import rasterio as rio
from torch.utils.data import Dataset


def channel_first(**kwargs):
    output = {}
    for k, v in kwargs.items():
        output[k] = torch.tensor(v.transpose(2, 0, 1) if v.ndim == 3 else v)
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

        # Remove the artifacts from masks
        return image, mask


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class InferenceDataset(Dataset):
    identity = rio.Affine(1, 0, 0, 0, 1, 0)

    def __init__(self, filename, sz, reduction):
        self.data = rio.open(
            filename.with_suffix('.tiff'),
            transform=self.identity,
            num_threads='all_cpus'
        )
        self.shape = self.data.shape
        self.reduction = reduction
        self.sz = reduction * sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in
        # the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        # mapping the loade region to the tile
        w = rio.Window.from_slices((p00, p01), (p10, p11))
        img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = \
            np.moveaxis(self.data.read([1, 2, 3], window=w), 0, -1)

        if self.reduction != 1:
            newshape = (self.sz // self.reduction, self.sz // self.reduction)
            img = cv2.resize(img, newshape, interpolation=cv2.INTER_AREA)

        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if self.is_saturated(img):
            # images with -1 will be skipped
            return img2tensor((img / 255.0 - self.mean) / self.std), -1

        return img2tensor((img / 255.0 - self.mean) / self.std), idx

    def is_saturated(self, img, s_th, p_th):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return (s > s_th).sum() <= p_th or img.sum() <= p_th
