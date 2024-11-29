import torch
from torch.utils.data.dataset import Dataset

import albumentations as A
import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


class CarvanaDataset(Dataset):
    """
    Load Carvana dataset.

    Args:
        image_directory (pathlib.Path): Path to the images' directory.
        mask_directory (pathlib.Path): Path to the masks' directory.
        transform (albumentations.Compose, optional): Transform to apply to the images and masks.
    """

    def __init__(
            self,
            image_directory: Path,
            mask_directory: Path,
            transform: Optional[A.Compose] = None,
    ) -> None:
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.images = os.listdir(image_directory)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, torch.Tensor]:
        image_path = os.path.join(self.image_directory, self.images[index])
        mask_path = os.path.join(self.mask_directory, self.images[index].replace(".jpg", "_mask.gif"))

        # Load images and masks
        image = np.array(Image.open(image_path).convert("RGB"))  # Shape: (H, W, 3)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # Shape: (H, W)
        mask = np.where(mask > 0, 1.0, 0.0)  # Ensure binary masks

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, mask
