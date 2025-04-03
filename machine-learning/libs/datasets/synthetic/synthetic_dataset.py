import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            phase: DatasetPhase = DatasetPhase.Train,
            use_trimap: bool = False,
            use_composition: bool = False,
    ) -> None:
        """
        Args:
            root_directory (str): Root directory of the dataset.
            transforms (transforms.Compose): Transform to apply to the data.
            phase (str): Phase of the dataset. Default: "train".
            use_trimap (bool): If True, use trimap images. Default: False.
            use_composition (bool): If True, use composition images. Default: False.
        """
        super().__init__()

        self.phase = phase
        self.use_trimap = use_trimap
        self.use_composition = use_composition
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

        if phase == DatasetPhase.Train:
            if use_trimap:
                self.trimap_paths = sorted([
                    os.path.join(base_directory, "trimaps", f)
                    for f in os.listdir(os.path.join(base_directory, "trimaps"))
                ])

            if use_composition:
                self.fg_paths = sorted([
                    os.path.join(base_directory, "fg", f)
                    for f in os.listdir(os.path.join(base_directory, "fg"))
                ])
                self.bg_paths = sorted([
                    os.path.join(base_directory, "bg", f)
                    for f in os.listdir(os.path.join(base_directory, "bg"))
                ])

    def _safe_listdir(self, path: str):
        return sorted([os.path.join(path, f) for f in os.listdir(path)]) if os.path.exists(path) else []

    def __len__(self):
        return len(self.alpha_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.images_paths[idx])
        alpha = cv2.imread(self.alpha_paths[idx], flags=0).astype(np.float32) / 255.0

        sample = {"image": image, "alpha": alpha}

        if self.phase == DatasetPhase.Train:
            if self.use_trimap:
                trimap = cv2.imread(self.trimap_paths[idx], flags=0).astype(np.float32) / 255.0
                sample["trimap"] = trimap

            if self.use_composition:
                fg = cv2.imread(self.fg_paths[idx])
                bg = cv2.imread(self.bg_paths[idx])
                sample["fg"] = fg
                sample["bg"] = bg

        return self.transforms(sample)


def debug_sample(sample: Dict[str, torch.Tensor]):
    image, alpha = sample["image"], sample["alpha"]

    # Assertions and value checks
    assert image.ndim == 3 and image.shape[0] == 3, "Image should be [3, H, W]"
    assert alpha.ndim == 3 and alpha.shape[0] == 1, "Alpha should be [1, H, W]"

    # Value sanity checks
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Alpha range: [{alpha.min().item():.3f}, {alpha.max().item():.3f}]")

    if "trimap" in sample:
        trimap = sample["trimap"]

        assert trimap.ndim == 3 and trimap.shape[0] == 1, "Trimap should be [1, H, W]"
        print(f"Trimap unique values: {torch.unique(trimap)}")

        sample_map = (trimap == 0.5).float()
        print(f"Sample map unique values: {torch.unique(sample_map)}")

    if "fg" in sample:
        fg = sample["fg"]

        assert fg.shape == image.shape, "FG must match image shape"
        print(f"FG range: [{fg.min().item():.3f}, {fg.max().item():.3f}]")

    if "bg" in sample:
        bg = sample["bg"]

        assert bg.shape == image.shape, "BG must match image shape"
        print(f"BG range: [{bg.min().item():.3f}, {bg.max().item():.3f}]")

    def denormalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
        std = torch.tensor(std).view(3, 1, 1).to(img.device)
        return img * std + mean

    # Prepare a list of items to visualize
    items = []

    # Image
    image_vis = denormalize(image).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    items.append(("Image", image_vis, "rgb"))

    # Alpha
    alpha_vis = alpha.squeeze().cpu().numpy()
    items.append(("Alpha", alpha_vis, "gray"))

    # Trimap
    if "trimap" in sample:
        trimap = sample["trimap"]
        trimap_vis = trimap.squeeze().cpu().numpy()
        items.append(("Trimap", trimap_vis, "gray"))

    # Foreground
    if "fg" in sample:
        fg = sample["fg"]
        fg_vis = denormalize(fg).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        items.append(("FG", fg_vis, "rgb"))

    # Background
    if "bg" in sample:
        bg = sample["bg"]
        bg_vis = denormalize(bg).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        items.append(("BG", bg_vis, "rgb"))

    # Create subplots based on the number of items to visualize
    num_items = len(items)
    fig, axes = plt.subplots(1, num_items, figsize=(6 * num_items, 5))

    for ax, (title, img, cmap) in zip(axes, items):
        ax.imshow(img, cmap=cmap if cmap == "gray" else None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix=ConfigurationSuffix.UNET)
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    # Create transforms for the training phase
    train_transforms = T.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(
            output_size=(224, 224),
            top_crop_ratio=0.4,
            low_threshold=0.1,
            high_threshold=0.9,
        ),
        RandomJitter(),
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = SyntheticDataset(
        root_directory=root_directory,
        transforms=train_transforms,
        phase=DatasetPhase.Train,
        use_trimap=True,
        use_composition=True,
    )
    sample = train_dataset[3]
    debug_sample(sample)
