from torch import Tensor
import torchvision.transforms.functional as F

import random
from PIL import Image
from typing import Callable, List, Tuple, Union
import numpy as np

Transform = Callable[
    [Image.Image, Image.Image],
    Tuple[Union[Image.Image, Tensor], Union[Image.Image, Tensor]]
]


class RandomHorizontalFlip(object):
    """
    Randomly flip the image and mask horizontally.
    """

    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Probability of the image and mask being flipped. Default: 0.5.
        """
        self.p = p

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        if random.random() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)

        return image, mask


class RandomSkyCrop(object):
    """
    Randomly crop the image and mask to a specified size ensuring that the crop
    contains at least a minimum fraction of non-zero (e.g., sky) pixels in the mask.
    """

    def __init__(self, size: Tuple[int, int], min_mask_ratio: float = 0.1, max_attempts: int = 10):
        """
        Args:
            size (Tuple[int, int]): Desired output size (height, width).
            min_mask_ratio (float): Minimum ratio of non-zero pixels in the cropped mask.
            max_attempts (int): Number of attempts to find a valid crop before giving up.
        """
        self.crop_height, self.crop_width = size
        self.min_mask_ratio = min_mask_ratio
        self.max_attempts = max_attempts

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        w, h = image.size

        if h < self.crop_height or w < self.crop_width:
            raise ValueError("Image is smaller than the crop size.")

        for _ in range(self.max_attempts):
            top = random.randint(0, h - self.crop_height)
            left = random.randint(0, w - self.crop_width)

            # Crop both image and mask
            image_crop = image.crop((left, top, left + self.crop_width, top + self.crop_height))
            mask_crop = mask.crop((left, top, left + self.crop_width, top + self.crop_height))

            # Convert mask crop to numpy array to compute the ratio of non-zero pixels
            mask_np = np.array(mask_crop)
            nonzero_ratio = np.count_nonzero(mask_np) / mask_np.size

            if nonzero_ratio >= self.min_mask_ratio:
                return image_crop, mask_crop

        # If no valid crop is found after max_attempts, return the last crop
        return image_crop, mask_crop


class Resize(object):
    """
    Resize the image and mask.
    """

    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size (int): Desired output size.
        """
        self.size = size

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        image = F.resize(image, list(self.size), interpolation=Image.BILINEAR)
        mask = F.resize(mask, list(self.size), interpolation=Image.NEAREST)

        return image, mask


class ToTensor(object):
    """
    Convert the image and mask to tensor.
    """

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)

        return image, mask


class Normalize(object):
    """
    Normalize the image.
    """

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        """
        Args:
            mean (Tuple[float, float, float]): Sequence of means for each channel.
            std (Tuple[float, float, float]): Sequence of standard deviations for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        image = F.normalize(image, mean=list(self.mean), std=list(self.std))

        return image, mask


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms: List[Transform]):
        """
        Args:
            transforms (List[Transform]): List of transforms to compose.
        """
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Tensor, Tensor]:
        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask
