import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from libs.datasets.transforms import RandomAffine, RandomCrop, RandomJitter, ToTensor, Normalize, OriginScale, \
    GenerateTrimap, GenerateFGBG, TopBiasedRandomCrop
from libs.configuration.configuration import get_configuration, ConfigurationMode

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data.
    """

    def __init__(
            self,
            root_directory: str,
            resolution: int = 512,
            phase: str = "train"
    ) -> None:
        """
        Args:
            root_directory (str): Root directory of the dataset.
            resolution (int): Size of the image. Default: 512.
            phase (str): Phase of the dataset. Default: "train".
        """
        super().__init__()

        base_directory = os.path.join(root_directory, phase)
        self.images_dir = os.path.join(base_directory, "images")
        self.masks_dir = os.path.join(base_directory, "masks")
        self.images = sorted(os.listdir(self.images_dir))
        self.alphas = sorted(os.listdir(self.masks_dir))

        self.transforms = {
            'train': transforms.Compose([
                RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                TopBiasedRandomCrop(output_size=(512, 512), vertical_bias_ratio=0.2),
                RandomJitter(),
                GenerateTrimap(),  # This generates the trimap from the ground truth alpha.
                GenerateFGBG(),  # This generates the fg and bg from the ground truth alpha.
                ToTensor(),
                Normalize()
            ]),
            'val': transforms.Compose([
                OriginScale(resolution),
                ToTensor(),
                Normalize()
            ]),
            'test': transforms.Compose([
                OriginScale(resolution),
                ToTensor(),
                Normalize()
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


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix="unet")
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    train_dataset = SyntheticDataset(
        root_directory=root_directory,
        resolution=config.scratch.resolution,
        phase="train",
    )
    sample = train_dataset[2]
    image, alpha, fg, bg, trimap = sample['image'], sample['alpha'], sample['fg'], sample['bg'], sample['trimap']

    # Print the shapes of the image, alpha mask, and trimap
    print(f"Train image shape: {image.shape}")  # torch.Size([3, 512, 512])
    print(f"Train alpha mask shape: {alpha.shape}")  # torch.Size([1, 512, 512])
    print(f"Train trimap shape: {trimap.shape}")  # torch.Size([1, 512, 512])
    print(f"Train fg mask shape: {fg.shape}")  # torch.Size([3, 512, 512])
    print(f"Train bg mask shape: {bg.shape}")  # torch.Size([3, 512, 512])


    def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # Ensure mean and std are tensors and on the same device as the image.
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        return image * std + mean


    # Create one figure with a custom gridspec layout.
    fig = plt.figure(figsize=(18, 12))
    # Use a grid with 2 rows and 6 columns.
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1])

    # Top row: two subplots (each spans 3 columns).
    ax_image = fig.add_subplot(gs[0, 0:3])
    ax_alpha = fig.add_subplot(gs[0, 3:6])

    # Bottom row: three subplots (each spans 2 columns).
    ax_trimap = fig.add_subplot(gs[1, 0:2])
    ax_fg = fig.add_subplot(gs[1, 2:4])
    ax_bg = fig.add_subplot(gs[1, 4:6])

    # Plot the image (denormalized).
    ax_image.imshow(denormalize(image).permute(1, 2, 0).clamp(0, 1))
    ax_image.set_title("RGB Image")
    ax_image.axis("off")

    # Plot the alpha mask.
    ax_alpha.imshow(alpha.squeeze(), cmap="gray")
    ax_alpha.set_title("Alpha Mask")
    ax_alpha.axis("off")

    # Plot the trimap.
    ax_trimap.imshow(trimap.squeeze(), cmap="gray")
    ax_trimap.set_title("Trimap")
    ax_trimap.axis("off")

    # Plot the foreground.
    ax_fg.imshow(denormalize(fg).permute(1, 2, 0).clamp(0, 1))
    ax_fg.set_title("Foreground")
    ax_fg.axis("off")

    # Plot the background.
    ax_bg.imshow(denormalize(bg).permute(1, 2, 0).clamp(0, 1))
    ax_bg.set_title("Background")
    ax_bg.axis("off")

    plt.tight_layout()
    plt.show()

    # Extra
    trimap[trimap < 85] = 0
    trimap[trimap > 170] = 1
    trimap[trimap >= 85] = 0.5
    sample_map = torch.zeros_like(trimap)
    sample_map[trimap == 0.5] = 1

    print(f"Train sample map shape: {sample_map.shape}")  # torch.Size([1, 512, 512])
    print(f"Train sample map unique values: {sample_map.unique()}")  # tensor([0., 1.])
