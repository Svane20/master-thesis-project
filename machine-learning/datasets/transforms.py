import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(resolution: int) -> A.Compose:
    """
    Get the train transforms with edge-focused augmentations.

    Applies spatial transforms (resize and flips) to both image and mask.
    Photometric transforms (CLAHE, Sharpen, BrightnessContrast) are applied only to the image.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        albumentations.Compose: Train transforms.
    """
    return A.Compose(
        [
            # Horizontal flip with a probability of 0.5.
            A.HorizontalFlip(p=0.5),

            # Random cropping of the image and mask with a probability of 0.5.
            A.RandomResizedCrop(size=(resolution, resolution), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),

            # Resize the image and mask to the target resolution.
            A.Resize(height=resolution, width=resolution),

            # Normalize the image and mask.
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),

            # Convert the image and mask to PyTorch tensors.
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )


def get_test_transforms(resolution: int) -> A.Compose:
    """
    Get the testing transforms.

    Applies spatial transforms to both image and mask. No photometric changes.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        albumentations.Compose: Testing transforms.
    """
    return A.Compose(
        [
            # Resize the image and mask to the target resolution.
            A.Resize(height=resolution, width=resolution),

            # Normalize the image and mask.
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),

            # Convert the image and mask to PyTorch tensors.
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )
