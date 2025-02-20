import torch
from torch.utils.data.dataset import Dataset

from albumentations import Compose
import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional


class SyntheticDataset(Dataset):
    """
    Load Synthetic dataset.

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
            file_list: Optional[List[str]] = None,
    ) -> None:
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transforms = transforms

        if file_list is not None:
            self.images = file_list
        else:
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
        # Load the image
        image_filename = self.images[index]
        image_path = os.path.join(self.image_directory, image_filename)

        # Load the mask
        mask_filename = image_filename.replace("Image", "SkyMask")
        mask_path = os.path.join(self.mask_directory, mask_filename)

        # Load the image and convert to RGB
        image = np.array(Image.open(image_path).convert("RGB"))

        # Load the mask as an RGBA image, then extract the alpha channel.
        mask_rgba = np.array(Image.open(mask_path).convert("RGBA"), dtype=np.float32)
        mask = mask_rgba[..., 3]  # extract alpha channel
        mask = mask / 255.0        # normalize to [0, 1]

        # Apply transforms if provided.
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert the image (HWC) to tensor (CHW)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Ensure mask has an explicit channel dimension.
        if isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            mask = torch.from_numpy(mask).float()
        elif isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Clamp the mask values to [0, 1]
        mask = mask.clamp(0, 1)

        return image, mask
