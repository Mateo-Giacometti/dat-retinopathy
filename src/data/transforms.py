"""Pipelines de aumentacion (Albumentations) segun propuesta: rotaciones 0-360,
espejados, brillo/contraste, escalado/recorte."""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=360, border_mode=0, fill=0, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.05),
                 border_mode=0, fill=0, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
