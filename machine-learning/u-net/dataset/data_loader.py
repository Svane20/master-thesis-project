from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from constants.directories import TRAIN_IMAGE_DIRECTORY, TRAIN_ANNOTATION_DIRECTORY
from dataset.ade_20k_dataset import ADE20KDataset


def create_train_data_loader() -> DataLoader:
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform masks into binary or multi-class tensors
    target_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Create Dataset and DataLoader instances
    dataset = ADE20KDataset(
        img_dir=TRAIN_IMAGE_DIRECTORY,
        mask_dir=TRAIN_ANNOTATION_DIRECTORY,
        transform=transform,
        target_transform=target_transform
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    return dataloader
