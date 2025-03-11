from ..transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, Transform


def get_train_transforms(resolution: int) -> Transform:
    """
    Get the training transforms.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        Transform: Training transforms.
    """
    return Compose([
        Resize((600, 600)),
        RandomCrop((resolution, resolution)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_test_transforms(resolution: int) -> Transform:
    """
    Get the testing transforms.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        Transform: Training transforms.
    """
    return Compose([
        Resize((resolution, resolution)),
        ToTensor(),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
