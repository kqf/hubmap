import albumentations as alb
from albumentations.pytorch import ToTensorV2

_mean = [0.485, 0.456, 0.406],
_std = [0.229, 0.224, 0.225]


def transform(train=True, mean=None, std=None):
    normalize = alb.Compose([
        alb.PadIfNeeded(256, 256),
        alb.Normalize(mean=mean or _mean, std=std or _std),
        ToTensorV2(),
    ])

    if not train:
        return normalize

    return alb.Compose([
        alb.PadIfNeeded(256, 256),
        alb.RandomCrop(224, 224),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        normalize,
    ])
