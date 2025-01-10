import torch
from torch.utils.data.dataset import Dataset

from albumentations import Compose
import numpy as np
import os
import cv2
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

        # Load the binary mask and convert it to a smooth alpha mask
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = _convert_binary_mask_to_smooth_alpha_mask(mask)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Add channel dimension to the mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask


def _convert_binary_mask_to_smooth_alpha_mask(
        binary_mask: np.ndarray,
        threshold_thresh: int = 1,
        threshold_max_value: int = 255,
        blur_kernel_size: Tuple[int, int] = (5, 5),
        blur_sigma: Tuple[float, float] = (0, 0),
) -> np.ndarray:
    """
    Convert a binary mask to a smooth alpha mask.

    Args:
        binary_mask (np.ndarray): Binary mask to convert.
        threshold_thresh (int): Threshold value for the binary mask.
        threshold_max_value (int): Maximum value for the threshold.
        blur_kernel_size (Tuple[int, int]): Kernel size for Gaussian blur.
        blur_sigma (Tuple[float, float]): Sigma values for Gaussian blur in x and y directions.

    Returns:
        np.ndarray: Smooth alpha mask.
    """
    # Set the threshold for the binary mask
    _, binary_mask = cv2.threshold(
        src=binary_mask,
        thresh=threshold_thresh,
        maxval=threshold_max_value,
        type=cv2.THRESH_BINARY
    )

    # Apply Gaussian blur to the binary mask
    blurred_mask = cv2.GaussianBlur(src=binary_mask, ksize=blur_kernel_size, sigmaX=blur_sigma[0], sigmaY=blur_sigma[1])

    # Normalize the blurred mask to [0, 1]
    alpha_mask = blurred_mask / 255.0

    # Ensure the alpha mask is in the range [0, 1]
    alpha_mask = np.clip(alpha_mask, a_min=0, a_max=1)

    return alpha_mask
