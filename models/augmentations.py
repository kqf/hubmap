import cv2
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
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.RandomRotate90(),
        alb.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.9,
            border_mode=cv2.BORDER_REFLECT),
        alb.OneOf([
            alb.OpticalDistortion(p=0.3),
            alb.GridDistortion(p=.1),
            alb.IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        alb.OneOf([
            alb.HueSaturationValue(10, 15, 10),
            alb.CLAHE(clip_limit=2),
            alb.RandomBrightnessContrast(),
        ], p=0.3),
        normalize,
    ])
