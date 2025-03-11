from torch import Tensor
import torchvision.transforms.functional as F

import random
from PIL import Image
from typing import Callable, List, Tuple, Union

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


class RandomCrop(object):
    """
    Randomly crop the image and mask to a specified size.
    """

    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size (Tuple[int, int]): Desired output size.
        """
        self.crop_height, self.crop_width = size

    def __call__(self, image: Image, mask: Image) -> Tuple[Image, Image]:
        w, h = image.size

        if h < self.crop_height or w < self.crop_width:
            raise ValueError("Image is smaller than the crop size.")

        top = random.randint(0, h - self.crop_height)
        left = random.randint(0, w - self.crop_width)

        image = image.crop((left, top, left + self.crop_width, top + self.crop_height))
        mask = mask.crop((left, top, left + self.crop_width, top + self.crop_height))

        return image, mask


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
        mask = F.resize(mask, list(self.size), interpolation=Image.BILINEAR)

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
