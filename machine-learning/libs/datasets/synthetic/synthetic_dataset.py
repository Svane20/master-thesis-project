from torch import Tensor
from torch.utils.data import Dataset

import os
import re
import logging
from PIL import Image

from ..transforms import Transform


class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(self, root_directory: str, transforms: Transform = None):
        """
        Args:
            root_directory (str): Path to the root directory.
            transforms (Transform, optional): Transforms to apply to the images and masks. Defaults to None.
        """
        self.images_dir = os.path.join(root_directory, "images")
        self.masks_dir = os.path.join(root_directory, "masks")

        try:
            image_files = os.listdir(self.images_dir)
        except Exception as e:
            logging.error(f"Error reading images directory {self.images_dir}: {e}")
            raise e

        try:
            mask_files = os.listdir(self.masks_dir)
        except Exception as e:
            logging.error(f"Error reading masks directory {self.masks_dir}: {e}")
            raise e

        # Regex to extract key: groups (timestamp, index)
        # Example:
        #   Image: 2025-02-21_10-50-50_Image_0.png  -> key: "2025-02-21_10-50-50_0"
        #   Mask:  2025-02-21_10-50-50_SkyMask_0.png -> key: "2025-02-21_10-50-50_0"
        pattern = re.compile(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_.*_(\d+)\.png")

        # Build dictionaries mapping key -> filename
        self.image_dict = {}
        for f in image_files:
            match = pattern.match(f)
            if match:
                key = f"{match.group(1)}_{match.group(2)}"
                self.image_dict[key] = f
            else:
                logging.warning(f"Image file {f} does not match the expected pattern.")

        self.mask_dict = {}
        for f in mask_files:
            match = pattern.match(f)
            if match:
                key = f"{match.group(1)}_{match.group(2)}"
                self.mask_dict[key] = f
            else:
                logging.warning(f"Mask file {f} does not match the expected pattern.")

        self.keys = sorted(set(self.image_dict.keys()) & set(self.mask_dict.keys()))
        if not self.keys:
            raise RuntimeError("No matching image-mask pairs found.")

        self.transforms = transforms

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        image_filename = self.image_dict[key]
        mask_filename = self.mask_dict[key]

        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logging.error(f"Error opening image file {image_path}: {e}")
            raise e

        try:
            mask_rgba = Image.open(mask_path).convert("RGBA")
        except Exception as e:
            logging.error(f"Error opening mask file {mask_path}: {e}")
            raise e

        # Preprocess mask by extracting the alpha channel
        alpha_channel = mask_rgba.getchannel("A")
        mask = alpha_channel

        if self.transforms:
            image, mask = self.transforms(image, mask)

            if isinstance(image, Tensor) and isinstance(mask, Tensor):
                if image.shape[-2:] != mask.shape[-2:]:
                    raise ValueError(
                        f"Transformed tensor dimensions do not match for key {key}: image shape {image.shape} vs. mask shape {mask.shape}"
                    )
            elif hasattr(image, "size") and hasattr(mask, "size"):
                if image.size != mask.size:
                    raise ValueError(
                        f"Transformed image sizes do not match for key {key}: image size {image.size} vs. mask size {mask.size}"
                    )

        return image, mask
