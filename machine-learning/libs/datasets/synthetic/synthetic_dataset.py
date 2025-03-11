from torch.utils.data import Dataset

import os
from PIL import Image
from typing import List
import numpy as np

from ..transforms import Transform


class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(self, root_directory: str, transforms: Transform = None):
        """
        Args:
            root_directory (str): Path to the root directory.
            transforms (List[Transform]): List of transforms to apply to the images and masks.
        """
        self.images_dir = os.path.join(root_directory, "images")
        self.masks_dir = os.path.join(root_directory, "masks")

        self.images = sorted(os.listdir(self.images_dir))
        self.masks = sorted(os.listdir(self.masks_dir))

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.masks[index])

        image = Image.open(image_path).convert("RGB")

        mask_rgba = Image.open(mask_path).convert("RGBA")
        mask_np = np.array(mask_rgba, dtype=np.float32)
        mask = mask_np[..., 3] / 255.0  # Normalize alpha to [0, 1]
        mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
