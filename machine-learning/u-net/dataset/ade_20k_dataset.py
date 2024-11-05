from torch.utils.data.dataset import Dataset

import os
from PIL import Image


class ADE20KDataset(Dataset):
    """
    Load ADE20K dataset.

    Args:
        img_dir (str): Path to the images' directory.
        mask_dir (str): Path to the masks' directory.
        transform (callable, optional): Optional transform to be applied on an image.
        target_transform (callable, optional): Optional transform to be applied on a mask.
    """

    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask
