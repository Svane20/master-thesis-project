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
            # Spatial transforms applied to both image and mask.
            A.HorizontalFlip(p=0.5),

            # Random cropping to the target resolution with 80% probability.
            A.RandomResizedCrop(size=(resolution, resolution), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=0.5),

            # Resize in case the crop doesn't exactly match your desired dimensions
            A.Resize(height=resolution, width=resolution),

            # Photometric transforms applied only to the image (they are image-only transforms).
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2.0, p=0.5),
                    A.Sharpen(alpha=(0.2, 0.5), p=0.5),
                ],
                p=0.5,
            ),

            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}  # ensure "mask" key is processed appropriately
    )


def get_val_transforms(resolution: int) -> A.Compose:
    """
    Get the validation transforms.

    Applies spatial transforms to both image and mask. No photometric changes.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        albumentations.Compose: Validation transforms.
    """
    return A.Compose(
        [
            A.Resize(resolution, resolution),
            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
            ToTensorV2(),
        ],
        additional_targets={'mask': 'mask'}
    )