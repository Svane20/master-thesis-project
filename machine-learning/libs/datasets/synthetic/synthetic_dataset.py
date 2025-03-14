import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

import os
from PIL import Image
from typing import Tuple

from ..transforms import Transform


class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(self, root_directory: str, transforms: Transform = None) -> None:
        """
        Args:
            root_directory (str): Path to the root directory.
            transforms (Transform, optional): Transforms to apply to the images and masks. Defaults to None.
        """
        self.images_dir = os.path.join(root_directory, "images")
        self.masks_dir = os.path.join(root_directory, "masks")
        self.image_files = sorted(os.listdir(self.images_dir))
        self.mask_files = sorted(os.listdir(self.masks_dir))

        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.image_files[index])
        mask_path = os.path.join(self.masks_dir, self.mask_files[index])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)
        else:
            image = to_tensor(image)
            mask = to_tensor(mask)

        return image, mask
