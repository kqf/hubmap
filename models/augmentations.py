import albumentations as alb
from albumentations.pytorch import ToTensorV2


def transform(train=True):
    normalize = alb.Compose([
        alb.PadIfNeeded(256, 256),
        alb.Normalize(  # TODO: Fix me
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
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
