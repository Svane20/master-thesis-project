import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from enum import Enum
from typing import Dict

from libs.configuration.configuration import get_configuration, ConfigurationMode, ConfigurationSuffix
from libs.datasets.transforms import RandomAffine, RandomJitter, ToTensor, Normalize, Rescale, TopBiasedRandomCrop

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class DatasetPhase(str, Enum):
    Train = "train"
    Val = "val"
    Test = "test"


class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(
            self,
            root_directory: str,
            transforms: T.Compose,
            phase: DatasetPhase = DatasetPhase.Train
    ) -> None:
        """
        Args:
            root_directory (str): Root directory of the dataset.
            transforms (transforms.Compose): Transform to apply to the data.
            phase (str): Phase of the dataset. Default: "train".
        """
        super().__init__()

        self.phase = phase
        self.transforms = transforms

        base_directory = os.path.join(root_directory, phase)
        self.images_paths = sorted([
            os.path.join(base_directory, "images", f)
            for f in os.listdir(os.path.join(base_directory, "images"))
        ])
        self.alpha_paths = sorted([
            os.path.join(base_directory, "masks", f)
            for f in os.listdir(os.path.join(base_directory, "masks"))
        ])

        if self.phase == DatasetPhase.Train:
            self.fg_paths = self._safe_listdir(os.path.join(base_directory, "fg"))
            self.bg_paths = self._safe_listdir(os.path.join(base_directory, "bg"))
            self.trimap_paths = self._safe_listdir(os.path.join(base_directory, "trimaps"))

    def _safe_listdir(self, path: str):
        return sorted([os.path.join(path, f) for f in os.listdir(path)]) if os.path.exists(path) else []

    def __len__(self):
        return len(self.alpha_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.images_paths[idx])
        alpha = cv2.imread(self.alpha_paths[idx], flags=0).astype(np.float32) / 255.0

        sample = {"image": image, "alpha": alpha, "fg": None, "bg": None, "trimap": None}

        if self.phase == DatasetPhase.Train:
            if idx < len(self.trimap_paths):
                trimap = cv2.imread(self.trimap_paths[idx], flags=0)
                if trimap is not None:
                    sample["trimap"] = trimap.astype(np.float32) / 255.0

            if idx < len(self.fg_paths):
                fg = cv2.imread(self.fg_paths[idx])
                sample["fg"] = fg

            if idx < len(self.bg_paths):
                bg = cv2.imread(self.bg_paths[idx])
                sample["bg"] = bg

        return self.transforms(sample)


def debug_sample(sample: Dict[str, torch.Tensor]):
    image, alpha, trimap = sample["image"], sample["alpha"], sample["trimap"]
    fg, bg = sample["fg"], sample["bg"]

    # Assertions and value checks
    assert image.ndim == 3 and image.shape[0] == 3, "Image should be [3, H, W]"
    assert alpha.ndim == 3 and alpha.shape[0] == 1, "Alpha should be [1, H, W]"
    assert trimap.ndim == 3 and trimap.shape[0] == 1, "Trimap should be [1, H, W]"
    assert fg.shape == image.shape, "FG must match image shape"
    assert bg.shape == image.shape, "BG must match image shape"

    # Value sanity checks
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Alpha range: [{alpha.min().item():.3f}, {alpha.max().item():.3f}]")
    print(f"Trimap unique values: {torch.unique(trimap)}")
    print(f"FG range: [{fg.min().item():.3f}, {fg.max().item():.3f}]")
    print(f"BG range: [{bg.min().item():.3f}, {bg.max().item():.3f}]")

    def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
        std = torch.tensor(std).view(3, 1, 1).to(img.device)
        return img * std + mean

    image_vis = denormalize(image).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    fg_vis = denormalize(fg).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    bg_vis = denormalize(bg).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    alpha_vis = alpha.squeeze().cpu().numpy()
    trimap_vis = trimap.squeeze().cpu().numpy()

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3)

    ax_image = fig.add_subplot(gs[0, 0])
    ax_image.imshow(image_vis)
    ax_image.set_title("Normalized RGB Image")
    ax_image.axis("off")

    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_alpha.imshow(alpha_vis, cmap="gray")
    ax_alpha.set_title("Alpha Mask (0=FG, 1=BG)")
    ax_alpha.axis("off")

    ax_trimap = fig.add_subplot(gs[0, 2])
    ax_trimap.imshow(trimap_vis, cmap="gray", vmin=0.0, vmax=1.0)
    ax_trimap.set_title("Trimap (0.0, 0.5, 1.0)")
    ax_trimap.axis("off")

    ax_fg = fig.add_subplot(gs[1, 0])
    ax_fg.imshow(fg_vis)
    ax_fg.set_title("Foreground (alpha < 0.5)")
    ax_fg.axis("off")

    ax_bg = fig.add_subplot(gs[1, 1])
    ax_bg.imshow(bg_vis)
    ax_bg.set_title("Background (alpha ≥ 0.5)")
    ax_bg.axis("off")

    ax_overlay = fig.add_subplot(gs[1, 2])
    overlay = alpha_vis[..., None] * bg_vis + (1 - alpha_vis[..., None]) * fg_vis
    ax_overlay.imshow(overlay)
    ax_overlay.set_title("Recomposited Overlay")
    ax_overlay.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix=ConfigurationSuffix.UNET)
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    # Create transforms for the training phase
    train_transforms = T.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(224, 224), top_crop_ratio=0.4),
        RandomJitter(),
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = SyntheticDataset(
        root_directory=root_directory,
        transforms=train_transforms,
        phase=DatasetPhase.Train,
    )
    sample = train_dataset[3]
    debug_sample(sample)
