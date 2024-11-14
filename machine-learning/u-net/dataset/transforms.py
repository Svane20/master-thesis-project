import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import os

from constants.outputs import IMAGE_SIZE

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """
    Get the train transforms with edge-focused augmentations.

    Args:
        image_size (Tuple[int, int]): Image size to resize to. Default is (224, 224).

    Returns:
        albumentations.Compose: Train transforms.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),

        # Flip augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Edge-preserving and detail-enhancing augmentations
        A.CLAHE(clip_limit=2.0, p=0.3),  # Apply CLAHE to enhance edges
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),  # Random sharpening
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

        # Blur and noise for robustness
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

        # Normalize and convert to tensor
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """
    Get the test transforms.

    Args:
        image_size (Tuple[int, int]): Image size to resize to. Default is (224, 224).

    Returns:
        albumentations.Compose: Test transforms.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),

        # Normalize and convert to tensor
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
