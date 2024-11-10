import torch
from torch.utils.data.dataset import Dataset

import os
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Callable


class CarvanaDataset(Dataset):
    """
    Load ADE20K dataset.

    Args:
        image_directory (pathlib.Path): Path to the images' directory.
        mask_directory (pathlib.Path): Path to the masks' directory.
        transform (callable, optional): Optional transform to be applied on an image.
        target_transform (callable, optional): Optional transform to be applied on a mask.
    """

    def __init__(
            self,
            image_directory: Path,
            mask_directory: Path,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.image_directory = image_directory
        self.mask_directory = mask_directory

        self.transform = transform
        self.target_transform = target_transform

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
        Get an item from the dataset.

        Args:
            index (int): Index of the item to get.

        Returns:
            Tuple[Image.Image, Image.Image]: Image and mask.
        """
        image_path = os.path.join(self.image_directory, self.images[index])
        mask_path = os.path.join(self.mask_directory, self.images[index].replace(".jpg", "_mask.gif"))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        # Ensure the mask is a float tensor
        mask = mask.float()

        return image, mask
