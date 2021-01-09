import sys
import cv2
import gc
import torch
import numpy as np
import pandas as pd
import rasterio as rio

from tqdm import tqdm
from pathlib import Path

MODELS = "/kaggle/input/hubmap-models/"
try:
    from models.modules import UNet
except ModuleNotFoundError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", MODELS])
    from models.modules import UNet


def read_model(path):
    model = UNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def rle_encode(img):
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


def nonpad(img, pad0, n0max, pad1, n1max, sz):
    return img[pad0 // 2:-(pad0 - pad0 // 2) if pad0 > 0 else n0max * sz,
               pad1 // 2:-(pad1 - pad1 // 2) if pad1 > 0 else n1max * sz]


class InferenceDataset(torch.utils.data.Dataset):
    mean = np.array([0.654599, 0.483866, 0.694284])
    std = np.array([0.151680, 0.235841, 0.131461])

    identity = rio.Affine(1, 0, 0, 0, 1, 0)

    def __init__(self, filename, sz, reduction):
        self.data = rio.open(
            filename,
            transform=self.identity,
            num_threads='all_cpus'
        )
        self.shape = self.data.shape
        self.reduction = reduction
        self.sz = reduction * sz
        self.sz_raw = sz
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
        w = rio.windows.Window.from_slices((p00, p01), (p10, p11))
        img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = \
            np.moveaxis(self.data.read([1, 2, 3], window=w), 0, -1)

        if self.reduction != 1:
            newshape = (self.sz // self.reduction, self.sz // self.reduction)
            img = cv2.resize(img, newshape, interpolation=cv2.INTER_AREA)

        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if self.is_saturated(img, s_th=40, p_th=200 * self.sz_raw // 256):
            # images with -1 will be skipped
            return img2tensor((img / 255.0 - self.mean) / self.std), -1

        return img2tensor((img / 255.0 - self.mean) / self.std), idx

    def is_saturated(self, img, s_th, p_th):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return (s > s_th).sum() <= p_th or img.sum() <= p_th


class InferenceModel:
    def __init__(self, models, dl, reduction):
        self.models = models
        self.dl = dl
        self.reduction = reduction

    def __call__(self, x, y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if ((y >= 0).sum() > 0):  # exclude empty images
            x = x[y >= 0].to(device)
            y = y[y >= 0]
            py = None
            for model in self.models:
                p = model(x)
                p = torch.sigmoid(p).detach()
                if py is None:
                    py = p
                else:
                    py += p

            py /= len(self.models)
            py = torch.nn.functional.upsample(
                py, scale_factor=self.reduction, mode="bilinear")
            py = py.permute(0, 2, 3, 1).float().cpu()

            batch_size = len(y)
            for i in range(batch_size):
                yield py[i], y[i]


def predict_masks(df, trainpath, models=[],
                  sz=256, reduction=4,
                  pthreshold=0.22, batch_size=32):
    preds, names = [], []
    for idx, (sample, *_) in tqdm(df.iterrows(), total=len(df)):
        tiff = (trainpath / sample).with_suffix(".tiff")
        ds = InferenceDataset(tiff, sz, reduction)

        # rasterio cannot be used with multiple workers
        dl = torch.utils.data.DataLoader(
            ds, batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=True
        )

        mp = InferenceModel(models, dl, reduction=reduction)

        # generate masks
        mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
        for x, y in iter(dl):
            for p, i in mp(x, y):
                mask[i.item()] = p.squeeze(-1) > pthreshold

        # reshape tiled masks into a single mask and crop padding
        mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz).\
            permute(0, 2, 1, 3).reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)

        mask = nonpad(mask, ds.pad0, ds.n0max, ds.pad1, ds.n1max, ds.sz)

        names.append(sample)
        preds.append(rle_encode(mask.numpy()))
        del mask, ds, dl
        gc.collect()
    return names, preds


def ensure_path(path, kernel_path="/kaggle/input/hubmap-kidney-segmentation/"):
    fpath = Path(path)
    if fpath.exists():
        return fpath
    return Path(kernel_path) / fpath.name


def _path(path, kernel_path="/kaggle/input/hubmap-kidney-segmentation/"):
    fpath = Path(path)
    if fpath.exists():
        return fpath
    return Path(kernel_path) / fpath


def main():
    df = pd.read_csv(ensure_path("data/sample_submission.csv"))

    # Run the inference
    trainpath = ensure_path("data/test")
    models = [read_model(_path("weights/params.pt", MODELS))]
    names, preds = predict_masks(df, trainpath, models=models)

    # Dump the predictions
    df = pd.DataFrame({'id': names, 'predicted': preds})
    df.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
