import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple
import os

from constants.outputs import IMAGE_SIZE

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def get_train_transforms(image_size: Tuple[int, int] = IMAGE_SIZE) -> A.Compose:
    """
    Get the train transforms.

    Args:
        image_size (Tuple[int, int]): Image size to resize to. Default is (224, 224).

    Returns:
        albumentations.Compose: Train transforms.
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])
