from torch.utils.data import Dataset

import os
from PIL import Image
from typing import List

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
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask
