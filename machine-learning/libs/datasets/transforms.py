import torch
import torchvision.transforms.functional as F

import cv2
import math
import numbers
import numpy as np
import random
from typing import Dict, Tuple
from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})

# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.random_interp = True

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


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

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image, alpha = sample['image'], sample['alpha']
        rows, cols, ch = image.shape

        params = self.get_params(
            self.degrees if max(rows, cols) >= 1024 else (0, 0),
            self.translate, self.scale, self.shear, self.flip, image.shape[:2]
        )
        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = np.array(self._get_inverse_affine_matrix(center, *params)).reshape((2, 3))

        def warp(data, interp=cv2.INTER_LINEAR):
            return cv2.warpAffine(data, M, (cols, rows), flags=interp + cv2.WARP_INVERSE_MAP)

        sample["image"] = warp(image, cv2.INTER_LINEAR)
        sample["alpha"] = warp(alpha, cv2.INTER_NEAREST)

        if "trimap" in sample and sample["trimap"] is not None:
            sample["trimap"] = warp(sample["trimap"], cv2.INTER_NEAREST)

        if "fg" in sample and sample["fg"] is not None:
            sample["fg"] = warp(sample["fg"], cv2.INTER_LINEAR)

        if "bg" in sample and sample["bg"] is not None:
            sample["bg"] = warp(sample["bg"], cv2.INTER_LINEAR)

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

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
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
        sat_mask = alpha > 0
        if np.count_nonzero(sat_mask) == 0:
            return sample_ori

        sat_bar = image[:, :, 1][sat_mask].mean()
        if np.isnan(sat_bar):
            return sample_ori

        sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
        sat = image[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        image[:, :, 1] = sat

        # Value noise
        val_mask = alpha > 0
        if np.count_nonzero(val_mask) == 0:
            return sample_ori

        val_bar = image[:, :, 2][val_mask].mean()
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

        hue_shift_float = float(hue_jitter)

        for key in ["fg", "bg"]:
            if key in sample and sample[key] is not None:
                # Convert to HSV
                # (No alpha check here, because 'fg' or 'bg' presumably is already a separate "layer")
                hsv = cv2.cvtColor(sample[key].astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)

                # Apply the same hue shift
                hsv[:, :, 0] = np.remainder(hsv[:, :, 0] + hue_shift_float, 360)

                # Apply sat_jitter
                s = hsv[:, :, 1]
                s = np.abs(s + sat_jitter)
                s[s > 1] = 2 - s[s > 1]
                hsv[:, :, 1] = s

                # Apply val_jitter
                v = hsv[:, :, 2]
                v = np.abs(v + val_jitter)
                v[v > 1] = 2 - v[v > 1]
                hsv[:, :, 2] = v

                # Convert back
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                sample[key] = bgr * 255

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if np.random.uniform(0, 1) < self.prob:
            def flip(data):
                return cv2.flip(data, 1)

            sample["image"] = flip(sample["image"])
            sample["alpha"] = flip(sample["alpha"])

            if "trimap" in sample and sample["trimap"] is not None:
                sample["trimap"] = flip(sample["trimap"])

            if "fg" in sample and sample["fg"] is not None:
                sample["fg"] = flip(sample["fg"])

            if "bg" in sample and sample["bg"] is not None:
                sample["bg"] = flip(sample["bg"])

        return sample


class TopBiasedRandomCrop(object):
    def __init__(
            self,
            output_size=(512, 512),
            top_crop_ratio=0.4,
            low_threshold: float = 0.0,
            high_threshold: float = 1.0,
    ):
        """
        Args:
            output_size (tuple): Desired crop size (H, W).
            top_crop_ratio (float): Proportion of the height to consider as top region (e.g., 0.4 = top 40%).
            low_threshold (float): Lower bound of the output image (e.g., 0.0).
            high_threshold (float): Upper bound of the output image (e.g., 1.0).
        """
        self.output_size = output_size
        self.top_crop_ratio = top_crop_ratio
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample['image']
        h, w, _ = image.shape
        crop_h, crop_w = self.output_size

        max_y = max(h - crop_h, 0)
        max_x = max(w - crop_w, 0)

        # Default: use top-biased random crop.
        start_y = random.randint(0, max(1, int(max_y * self.top_crop_ratio))) if max_y > 0 else 0
        start_x = random.randint(0, max_x) if max_x > 0 else 0

        # If alpha exists, try to focus on the intermediate region.
        if "alpha" in sample and sample["alpha"] is not None:
            alpha = sample["alpha"]

            # Define thresholds for intermediate values (you can adjust these)
            intermediate_mask = (alpha > self.low_threshold) & (alpha < self.high_threshold)

            if np.any(intermediate_mask):
                # Get bounding box coordinates of the intermediate area
                coords = np.argwhere(intermediate_mask)
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                # Compute the center of the bounding box
                center_y = (y_min + y_max) // 2
                center_x = (x_min + x_max) // 2

                # Set start positions so that the crop is centered around the intermediate region,
                # ensuring they fall within valid limits.
                start_y = max(0, min(center_y - crop_h // 2, max_y))
                start_x = max(0, min(center_x - crop_w // 2, max_x))

        # Apply crop to image and alpha.
        sample["image"] = image[start_y:start_y + crop_h, start_x:start_x + crop_w, :]
        if "alpha" in sample:
            sample["alpha"] = sample["alpha"][start_y:start_y + crop_h, start_x:start_x + crop_w]

        if "trimap" in sample and sample["trimap"] is not None:
            sample["trimap"] = sample["trimap"][start_y:start_y + crop_h, start_x:start_x + crop_w]
        if "fg" in sample and sample["fg"] is not None:
            sample["fg"] = sample["fg"][start_y:start_y + crop_h, start_x:start_x + crop_w, :]
        if "bg" in sample and sample["bg"] is not None:
            sample["bg"] = sample["bg"][start_y:start_y + crop_h, start_x:start_x + crop_w, :]

        return sample


class RandomCrop(object):
    def __init__(self, output_size=(512, 512)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = self.output_size[0] // 2

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image, alpha = sample['image'], sample['alpha']
        h, w, _ = image.shape

        if "trimap" in sample and sample["trimap"] is not None:
            trimap = sample["trimap"]
            # Resize logic if needed
            if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
                ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
                while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                    image = cv2.resize(image, (int(w * ratio), int(h * ratio)),
                                       interpolation=_maybe_random_interp(cv2.INTER_NEAREST))
                    alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                       interpolation=_maybe_random_interp(cv2.INTER_NEAREST))
                    trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)

                    if "fg" in sample and sample["fg"] is not None:
                        sample["fg"] = cv2.resize(sample["fg"], (int(w * ratio), int(h * ratio)),
                                                  interpolation=cv2.INTER_LINEAR)
                    if "bg" in sample and sample["bg"] is not None:
                        sample["bg"] = cv2.resize(sample["bg"], (int(w * ratio), int(h * ratio)),
                                                  interpolation=cv2.INTER_LINEAR)

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

            y0, x0 = left_top
        else:
            # Simple random crop if no trimap
            if w < self.output_size[1] or h < self.output_size[0]:
                y0 = (h - self.output_size[0]) // 2
                x0 = (w - self.output_size[1]) // 2
            else:
                y0 = random.randint(0, h - self.output_size[0])
                x0 = random.randint(0, w - self.output_size[1])

        y1, x1 = y0 + self.output_size[0], x0 + self.output_size[1]

        sample["image"] = image[y0:y1, x0:x1, :]
        sample["alpha"] = alpha[y0:y1, x0:x1]

        if "trimap" in sample and sample["trimap"] is not None:
            sample["trimap"] = sample["trimap"][y0:y1, x0:x1]

        if "fg" in sample and sample["fg"] is not None:
            sample["fg"] = sample["fg"][y0:y1, x0:x1, :]

        if "bg" in sample and sample["bg"] is not None:
            sample["bg"] = sample["bg"][y0:y1, x0:x1, :]

        return sample


class GenerateTrimap(object):
    """
    Generates a more consistent trimap from the ground truth alpha.
    Uses erosion to define definite fg/bg and leaves wider unknown areas.
    """

    def __init__(self, min_width: int = 5, max_width: int = 15):
        self.min_width = min_width
        self.max_width = max_width
        self.erosion_kernels = [None] + [
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            for size in range(1, 30)
        ]

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        original_alpha = sample["alpha"]
        h, w = original_alpha.shape

        alpha = cv2.resize(original_alpha, (640, 640), interpolation=cv2.INTER_LANCZOS4)

        fg_width = np.random.randint(self.min_width, self.max_width)
        bg_width = np.random.randint(self.min_width, self.max_width)

        fg_mask = (alpha >= 0.9).astype(np.uint8)
        bg_mask = (alpha <= 0.1).astype(np.uint8)

        fg_eroded = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_eroded = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha, dtype=np.uint8) * 128
        trimap[fg_eroded == 1] = 255
        trimap[bg_eroded == 1] = 0

        trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
        sample["trimap"] = trimap

        return sample


class GenerateFGBG(object):
    """
    Generates soft foreground and background images using the alpha matte.
    For sky replacement:
      - Foreground (building): (1 - alpha)
      - Background (sky): alpha
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if "fg" in sample and "bg" in sample:
            return sample

        if "image" not in sample or "alpha" not in sample:
            raise ValueError("Sample must contain 'image' and 'alpha' to generate foreground and background.")

        image = sample["image"]
        alpha = sample["alpha"]

        # Ensure alpha is float32 in range [0, 1]
        if alpha.dtype != np.float32:
            alpha = alpha.astype(np.float32) / 255.0

        # Expand dims to match image shape: H x W x 1
        if alpha.ndim == 2:
            alpha = alpha[..., None]

        fg = image * (1.0 - alpha)  # Building (where alpha ≈ 0)
        bg = image * alpha  # Sky (where alpha ≈ 1)

        sample["fg"] = fg
        sample["bg"] = bg
        return sample


class OriginScale(object):
    def __init__(self, output_size=512):
        self.output_size = output_size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        desired_size = self.output_size
        h, w, c = sample['image'].shape

        # Pad if necessary when image is smaller than desired size.
        pad_h = max(0, desired_size - h)
        pad_w = max(0, desired_size - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        def pad_if_needed(data, is_color=True):
            if is_color:
                return np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
            else:
                return np.pad(data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")

        sample['image'] = pad_if_needed(sample['image'], is_color=True)
        if 'alpha' in sample and sample['alpha'] is not None:
            sample['alpha'] = pad_if_needed(sample['alpha'], is_color=False)
        if 'trimap' in sample and sample['trimap'] is not None:
            sample['trimap'] = pad_if_needed(sample['trimap'], is_color=False)
        if 'fg' in sample and sample['fg'] is not None:
            sample['fg'] = pad_if_needed(sample['fg'], is_color=True)
        if 'bg' in sample and sample['bg'] is not None:
            sample['bg'] = pad_if_needed(sample['bg'], is_color=True)

        # Recalculate shape after padding
        h, w, _ = sample['image'].shape
        start_y = (h - desired_size) // 2
        start_x = (w - desired_size) // 2

        def crop_center(data, is_color=True):
            if is_color:
                return data[start_y:start_y + desired_size, start_x:start_x + desired_size, :]
            else:
                return data[start_y:start_y + desired_size, start_x:start_x + desired_size]

        sample['image'] = crop_center(sample['image'], is_color=True)
        if 'alpha' in sample and sample['alpha'] is not None:
            sample['alpha'] = crop_center(sample['alpha'], is_color=False)
        if 'trimap' in sample and sample['trimap'] is not None:
            sample['trimap'] = crop_center(sample['trimap'], is_color=False)
        if 'fg' in sample and sample['fg'] is not None:
            sample['fg'] = crop_center(sample['fg'], is_color=True)
        if 'bg' in sample and sample['bg'] is not None:
            sample['bg'] = crop_center(sample['bg'], is_color=True)

        return sample


class Resize(object):
    """
    Resize the image (and alpha, and optionally trimap, fg, bg) to a fixed size.
    """

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        sample["image"] = cv2.resize(sample["image"], self.size, interpolation=cv2.INTER_LINEAR)

        # Optional fields
        if "alpha" in sample and sample["alpha"] is not None:
            sample["alpha"] = cv2.resize(sample["alpha"], self.size, interpolation=cv2.INTER_NEAREST)

        if "trimap" in sample and sample["trimap"] is not None:
            sample["trimap"] = cv2.resize(sample["trimap"], self.size, interpolation=cv2.INTER_NEAREST)

        if "fg" in sample and sample["fg"] is not None:
            sample["fg"] = cv2.resize(sample["fg"], self.size, interpolation=cv2.INTER_LINEAR)

        if "bg" in sample and sample["bg"] is not None:
            sample["bg"] = cv2.resize(sample["bg"], self.size, interpolation=cv2.INTER_LINEAR)

        return sample


class ToTensor(object):
    """
    Convert ndarrays in samples to Tensors. Assumes input is still in [0, 255] uint8.
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        if "image" in sample and sample["image"] is not None:
            image = sample["image"][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)
            sample["image"] = torch.from_numpy(image)

        if "alpha" in sample and sample["alpha"] is not None:
            alpha = np.clip(sample["alpha"], 0, 1)
            if alpha.ndim == 2:
                alpha = np.expand_dims(alpha, axis=0)
            sample["alpha"] = torch.from_numpy(alpha.astype(np.float32))

        if "trimap" in sample and sample["trimap"] is not None:
            trimap = torch.from_numpy(sample["trimap"]).float()
            if trimap.max() > 1.0:
                trimap = trimap / 255.0

            unique_vals = torch.unique(trimap)
            if not set(unique_vals.tolist()).issubset({0.0, 0.5, 1.0}):
                trimap = torch.where(trimap < 0.25, 0.0, torch.where(trimap > 0.75, 1.0, 0.5))

            if trimap.ndim == 2:
                trimap = trimap.unsqueeze(0)

            sample["trimap"] = trimap

        for key in ["fg", "bg"]:
            if key in sample and sample[key] is not None:
                tensor = sample[key][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32)
                sample[key] = torch.from_numpy(tensor)

        return sample


class Rescale(object):
    """
    Rescale image pixels by a fixed factor (e.g., 1/255).
    """

    def __init__(self, scale=1 / 255.0) -> None:
        self.scale = scale

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in ["image", "fg", "bg"]:
            if key in sample and sample[key] is not None:
                sample[key] = sample[key] * self.scale

        return sample


class Normalize(object):
    """
    Normalize the image.
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key in ["image", "fg", "bg"]:
            if key in sample and sample[key] is not None:
                sample[key] = F.normalize(sample[key], self.mean, self.std, self.inplace)

        return sample


def _maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp
