import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as transforms

import os
from PIL import Image
from typing import Tuple, Dict
import cv2
import numpy as np
import random
import math
import numbers
from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})

# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.random_interp = True

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from libs.configuration.configuration import get_configuration, ConfigurationMode
from libs.datasets.transforms import Transform

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class OriginScale(object):
    def __init__(self, output_size=512):
        self.output_size = output_size

    def __call__(self, sample):
        desired_size = self.output_size
        h, w, c = sample['image'].shape

        # Pad if necessary when image is smaller than desired size.
        pad_h = max(0, desired_size - h)
        pad_w = max(0, desired_size - w)
        if pad_h > 0 or pad_w > 0:
            # Pad evenly on both sides.
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            sample['image'] = np.pad(
                sample['image'],
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="reflect"
            )
            if 'alpha' in sample:
                sample['alpha'] = np.pad(
                    sample['alpha'],
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="reflect"
                )
            # Update h, w after padding
            h, w, _ = sample['image'].shape

        # Now, perform a center crop to the desired_size x desired_size.
        start_y = (h - desired_size) // 2
        start_x = (w - desired_size) // 2
        sample['image'] = sample['image'][start_y:start_y + desired_size, start_x:start_x + desired_size, :]
        if 'alpha' in sample:
            sample['alpha'] = sample['alpha'][start_y:start_y + desired_size, start_x:start_x + desired_size]
        if 'trimap' in sample:
            sample['trimap'] = sample['trimap'][start_y:start_y + desired_size, start_x:start_x + desired_size]

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int32) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        rows, cols, ch = image.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, image.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, image.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        image = cv2.warpAffine(image, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['image'], sample['alpha'] = image, alpha

        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        sample_ori = sample.copy()
        image, alpha = sample['image'], sample['alpha']

        # if alpha is all 0 skip
        if np.all(alpha == 0):
            return sample_ori

        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        image = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)

        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        image[:, :, 0] = np.remainder(image[:, :, 0].astype(np.float32) + hue_jitter, 360)

        # Saturation noise
        sat_bar = image[:, :, 1][alpha > 0].mean()
        if np.isnan(sat_bar):
            return sample_ori

        sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
        sat = image[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        image[:, :, 1] = sat

        # Value noise
        val_bar = image[:, :, 2][alpha > 0].mean()
        if np.isnan(val_bar):
            return sample_ori

        val_jitter = np.random.rand() * (1.1 - val_bar) / 5 - (1.1 - val_bar) / 10
        val = image[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        image[:, :, 2] = val
        # convert back to BGR space
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        sample['image'] = image * 255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        image, alpha = sample['image'], sample['alpha']
        if np.random.uniform(0, 1) < self.prob:
            image = cv2.flip(image, 1)
            alpha = cv2.flip(alpha, 1)
        sample['image'], sample['alpha'] = image, alpha

        return sample


class RandomCrop(object):
    """
    Randomly crop the sample to the desired output size.
    If a trimap exists, use it for guiding the crop; otherwise, perform a simple random crop.
    """

    def __init__(self, output_size=(512, 512)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = self.output_size[0] // 2

    def __call__(self, sample):
        if "trimap" in sample:
            image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']
            h, w = trimap.shape
            # Resize if needed.
            if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
                ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
                while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                    image = cv2.resize(image, (int(w * ratio), int(h * ratio)),
                                       interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                    alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                       interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                    trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                    h, w = trimap.shape
            small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            unknown_list = list(zip(*np.where(small_trimap[self.margin // 4:(h - self.margin) // 4,
                                              self.margin // 4:(w - self.margin) // 4] == 128)))
            unknown_num = len(unknown_list)
            if unknown_num < 10:
                left_top = (np.random.randint(0, h - self.output_size[0] + 1),
                            np.random.randint(0, w - self.output_size[1] + 1))
            else:
                idx = np.random.randint(unknown_num)
                left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)
            image_crop = image[left_top[0]:left_top[0] + self.output_size[0],
                         left_top[1]:left_top[1] + self.output_size[1], :]
            alpha_crop = alpha[left_top[0]:left_top[0] + self.output_size[0],
                         left_top[1]:left_top[1] + self.output_size[1]]
            trimap_crop = trimap[left_top[0]:left_top[0] + self.output_size[0],
                          left_top[1]:left_top[1] + self.output_size[1]]
            sample.update({'image': image_crop, 'alpha': alpha_crop, 'trimap': trimap_crop})
        else:
            # If no trimap is present, perform a simple random crop.
            image, alpha = sample['image'], sample['alpha']
            h, w, _ = image.shape
            if w < self.output_size[1] or h < self.output_size[0]:
                start_y = (h - self.output_size[0]) // 2
                start_x = (w - self.output_size[1]) // 2
            else:
                start_y = random.randint(0, h - self.output_size[0])
                start_x = random.randint(0, w - self.output_size[1])
            image_crop = image[start_y:start_y + self.output_size[0], start_x:start_x + self.output_size[1], :]
            alpha_crop = alpha[start_y:start_y + self.output_size[0], start_x:start_x + self.output_size[1]]
            sample.update({'image': image_crop, 'alpha': alpha_crop})
        return sample


class GenerateTrimap(object):
    """
    Generates a trimap from the ground truth alpha.
    """

    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in
                                         range(1, 30)]

    def __call__(self, sample: Dict[str, np.ndarray]):
        original_alpha = sample["alpha"]
        h, w = original_alpha.shape

        # Resize alpha for consistent processing
        alpha = cv2.resize(original_alpha, (640, 640), interpolation=cv2.INTER_LANCZOS4)

        ### generate trimap
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int32).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int32).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        # Create trimap: 255 for definite foreground, 0 for background, 128 for unknown region.
        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        # Resize trimap back to original alpha dimensions
        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        return sample


class ToTensor(object):
    """
    Convert ndarrays in samples to Tensors with normalization.
    """

    def __init__(self, phase="test"):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.FloatTensor]:
        # Convert BGR to RGB by reversing channel order.
        image, alpha = sample["image"][:, :, ::-1], sample["alpha"]

        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1

        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha, axis=0).astype(np.float32)
        image /= 255.0

        if self.phase == "train" and "trimap" in sample:
            sample["trimap"] = torch.from_numpy(sample["trimap"]).to(torch.long)

        sample['image'] = torch.from_numpy(image)
        sample['alpha'] = torch.from_numpy(alpha)
        return sample


class SyntheticDatasetNew(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(self, root_directory: str, phase: str = "train") -> None:
        super().__init__()

        base_directory = os.path.join(root_directory, phase)
        self.images_dir = os.path.join(base_directory, "images")
        self.masks_dir = os.path.join(base_directory, "masks")
        self.images = sorted(os.listdir(self.images_dir))
        self.alphas = sorted(os.listdir(self.masks_dir))

        self.transforms = {
            'train': transforms.Compose([
                RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                GenerateTrimap(),
                RandomCrop((512, 512)),
                RandomJitter(),
                ToTensor()
            ]),
            'val': transforms.Compose([
                OriginScale(),
                ToTensor()
            ]),
            'test': transforms.Compose([
                OriginScale(),
                ToTensor()
            ])
        }[phase]

    def __len__(self):
        return len(self.alphas)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(image_path)

        alpha_path = os.path.join(self.masks_dir, self.alphas[idx])
        alpha = cv2.imread(alpha_path, 0).astype(np.float32) / 255.0

        return self.transforms({'image': image, 'alpha': alpha})


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


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix="unet")
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    train_dataset = SyntheticDatasetNew(
        root_directory=root_directory,
        phase="train",
    )
    train_sample = train_dataset[2]
    image, alpha, trimap = train_sample["image"], train_sample["alpha"], train_sample["trimap"]

    # Print the shapes of the image, alpha mask, and trimap
    print(f"Image shape: {image.shape}")
    print(f"Alpha mask shape: {alpha.shape}")
    print(f"Trimap shape: {trimap.shape}")

    # Visualize the image and alpha mask
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    # Display the RGB image.
    axs[0].imshow(image.permute(1, 2, 0))
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    # Display the alpha mask.
    axs[1].imshow(alpha.squeeze(), cmap="gray")
    axs[1].set_title("Alpha Mask")
    axs[1].axis("off")

    # Display the trimap.
    axs[2].imshow(trimap.squeeze(), cmap="gray")
    axs[2].set_title("Trimap")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
