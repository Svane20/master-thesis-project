from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

from constants.directories import DATA_DIRECTORY


def get_dataset(train: bool, transform: transforms.Compose) -> Dataset:
    """
    Get FashionMNIST dataset.

    Args:
        train (bool): If True, returns the training dataset; otherwise, returns the test dataset.
        transform (transforms.Compose): Transform to apply to the data.

    Returns:
        Dataset: The requested dataset.
    """
    return FashionMNIST(
        root=DATA_DIRECTORY,
        train=train,
        download=True,
        transform=transform
    )
