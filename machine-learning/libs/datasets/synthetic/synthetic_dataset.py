import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from enum import Enum

from libs.configuration.configuration import get_configuration, ConfigurationMode, ConfigurationSuffix
from libs.datasets.transforms import RandomAffine, TopBiasedRandomCrop, RandomJitter, GenerateTrimap, ToTensor, \
    Normalize, Rescale, GenerateFGBG

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

        base_directory = os.path.join(root_directory, phase)
        self.images_dir = os.path.join(base_directory, "images")
        self.alpha_dir = os.path.join(base_directory, "masks")
        self.images = sorted(os.listdir(self.images_dir))
        self.alphas = sorted(os.listdir(self.alpha_dir))

        self.transforms = transforms

    def __len__(self):
        return len(self.alphas)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(image_path)

        alpha_path = os.path.join(self.alpha_dir, self.alphas[idx])
        alpha = cv2.imread(alpha_path, flags=0).astype(np.float32) / 255.0

        return self.transforms({'image': image, 'alpha': alpha})


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix=ConfigurationSuffix.UNET)
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    # Create transforms for the training phase
    train_transforms = T.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(224, 224), vertical_bias_ratio=0.2),
        RandomJitter(),
        GenerateTrimap(),
        GenerateFGBG(),
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_dataset = SyntheticDataset(
        root_directory=root_directory,
        transforms=train_transforms,
        phase=DatasetPhase.Train,
    )
    sample = train_dataset[0]
    image, alpha, trimap = sample["image"], sample["alpha"], sample["trimap"]
    fg, bg = sample["fg"], sample["bg"]

    # Print the shapes of the image, alpha mask, and trimap
    print(f"Train image shape: {image.shape}")  # torch.Size([3, 224, 224])
    print(f"Train alpha mask shape: {alpha.shape}")  # torch.Size([1, 224, 224])
    print(f"Train trimap shape: {trimap.shape}")  # torch.Size([1, 224, 224])
    print(f"Train fg shape: {fg.shape}")  # torch.Size([3, 224, 224])
    print(f"Train bg shape: {bg.shape}")  # torch.Size([3, 224, 224])


    def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        return image * std + mean


    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3)

    ax_image = fig.add_subplot(gs[0, 0])
    ax_image.imshow(denormalize(image).permute(1, 2, 0).clamp(0, 1).cpu())
    ax_image.set_title("RGB Image")
    ax_image.axis("off")

    ax_alpha = fig.add_subplot(gs[0, 1])
    ax_alpha.imshow(alpha.squeeze().cpu(), cmap="gray")
    ax_alpha.set_title("Alpha Mask")
    ax_alpha.axis("off")

    ax_trimap = fig.add_subplot(gs[0, 2])
    ax_trimap.imshow(trimap.squeeze().cpu(), cmap="gray")
    ax_trimap.set_title("Trimap")
    ax_trimap.axis("off")

    ax_fg = fig.add_subplot(gs[1, 0])
    ax_fg.imshow(denormalize(fg).permute(1, 2, 0).clamp(0, 1).cpu())
    ax_fg.set_title("Foreground (alpha < 0.5)")
    ax_fg.axis("off")

    ax_bg = fig.add_subplot(gs[1, 1])
    ax_bg.imshow(denormalize(bg).permute(1, 2, 0).clamp(0, 1).cpu())
    ax_bg.set_title("Background (alpha ≥ 0.5)")
    ax_bg.axis("off")

    plt.tight_layout()
    plt.show()
