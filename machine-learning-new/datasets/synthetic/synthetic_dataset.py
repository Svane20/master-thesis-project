import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from albumentations import Compose


class SyntheticDataset(Dataset):
    """
    Load Synthetic dataset.

    Args:
        image_directory (Path): Path to the images' directory.
        mask_directory (Path): Path to the masks' directory.
        transforms (Compose, optional): Albumentations transforms to apply to the image and mask.
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
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load the image
        image_filename = self.images[index]
        image_path = os.path.join(self.image_directory, image_filename)

        # Load the mask (assuming mask filename is derived from image filename)
        mask_filename = image_filename.replace("Image", "SkyMask")
        mask_path = os.path.join(self.mask_directory, mask_filename)

        # Load the image (RGB)
        image = np.array(Image.open(image_path).convert("RGB"))

        # Load the mask as RGBA, extract the alpha channel, and normalize to [0, 1]
        mask_rgba = np.array(Image.open(mask_path).convert("RGBA"), dtype=np.float32)
        mask = mask_rgba[..., 3]  # extract alpha channel
        mask = mask / 255.0

        # Apply transforms if provided.
        if self.transforms:
            # The transform will process "image" and "mask" separately based on the type of transform.
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert image from HWC to CHW tensor if needed.
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
