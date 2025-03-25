import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
import cv2
import numpy as np

from libs.datasets.transforms import RandomAffine, RandomCrop, RandomJitter, ToTensor, Normalize, OriginScale
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
                RandomCrop((resolution, resolution)),
                RandomJitter(),
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

        sample = self.transforms({'image': image, 'alpha': alpha})

        return sample["image"], sample["alpha"]


if __name__ == "__main__":
    config = get_configuration(ConfigurationMode.Training, suffix="unet")
    root_directory = os.path.join(config.dataset.root, config.dataset.name)

    train_dataset = SyntheticDataset(
        root_directory=root_directory,
        resolution=config.scratch.resolution,
        phase="train",
    )
    train_image, train_alpha = train_dataset[2]

    # Print the shapes of the image, alpha mask, and trimap
    print(f"Train image shape: {train_image.shape}")  # torch.Size([3, 512, 512])
    print(f"Train alpha mask shape: {train_alpha.shape}")  # torch.Size([1, 512, 512])

    val_dataset = SyntheticDataset(
        root_directory=root_directory,
        resolution=config.scratch.resolution,
        phase="val",
    )
    val_image, val_alpha = val_dataset[1]
    print(f"Val image shape: {val_image.shape}")  # torch.Size([3, 512, 512])
    print(f"Val alpha mask shape: {val_alpha.shape}\n")  # torch.Size([1, 512, 512])

    test_dataset = SyntheticDataset(
        root_directory=root_directory,
        resolution=config.scratch.resolution,
        phase="test",
    )
    test_image, test_alpha = test_dataset[2]
    print(f"Test image shape: {test_image.shape}")  # torch.Size([3, 512, 512])
    print(f"Test alpha mask shape: {test_alpha.shape}")  # torch.Size([1, 512, 512])


    def denormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # Ensure mean and std are tensors and on the same device as the image.
        mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(std).view(3, 1, 1).to(image.device)
        return image * std + mean


    # Visualize the train image, alpha mask, and trimap
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Display the RGB image.
    axs[0].imshow(denormalize(train_image).permute(1, 2, 0).clamp(0, 1))
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    # Display the alpha mask.
    axs[1].imshow(train_alpha.squeeze(), cmap="gray")
    axs[1].set_title("Alpha Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Visualize the val image and alpha mask
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # Display the RGB image.
    axs[0].imshow(denormalize(val_image).permute(1, 2, 0).clamp(0, 1))
    axs[0].set_title("RGB Image")
    axs[0].axis("off")

    # Display the alpha mask.
    axs[1].imshow(val_alpha.squeeze(), cmap="gray")
    axs[1].set_title("Alpha Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
