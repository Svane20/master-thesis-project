from libs.datasets.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, \
    Transform, RandomCrop

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms(resolution: int, resize: int = 1000) -> Transform:
    """
    Get the training transforms.

    Args:
        resolution (int): Resolution of the image.
        resize (int): Resize the image to this before cropping. Default: 1000.

    Returns:
        Transform: Training transforms.
    """
    return Compose([
        Resize(size=(resize, resize)),
        RandomCrop(size=(resolution, resolution)),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])


def get_val_transforms(resolution: int) -> Transform:
    """
    Get the validation transforms.

    Args:
        resolution (int): Resolution of the image.

    Returns:
        Transform: Training transforms.
    """
    return Compose([
        Resize((resolution, resolution)),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
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
        Normalize(mean=MEAN, std=STD)
    ])
