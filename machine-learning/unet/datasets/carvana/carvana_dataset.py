import torch
from torch.utils.data.dataset import Dataset

from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


class CarvanaDataset(Dataset):
    """
    Load Carvana dataset.

    Args:
        image_directory (pathlib.Path): Path to the images' directory.
        mask_directory (pathlib.Path): Path to the masks' directory.
        transforms (albumentations.Compose, optional): Transform to apply to the images and masks.
    """

    def __init__(
            self,
            image_directory: Path,
            mask_directory: Path,
            transforms: Optional[Compose] = None,
    ) -> None:
        self.image_directory = image_directory
        self.mask_directory = mask_directory

        self.transforms = transforms

        self.images = os.listdir(image_directory)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """

        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the image and mask at the specified index.

        Args:
            index (int): Index of the image and mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and mask tensors.
        """
        image_path = os.path.join(self.image_directory, self.images[index])
        mask_path = os.path.join(self.mask_directory, self.images[index].replace(".jpg", "_mask.gif"))

        # Load the image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Load the mask and normalize to [0, 1]
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = mask / 255.0

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension to the mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask
