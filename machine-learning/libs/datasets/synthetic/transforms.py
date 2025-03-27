import torchvision.transforms as transforms

from ...datasets.transforms import Resize, ToTensor, Normalize, OriginScale, RandomAffine, TopBiasedRandomCrop, \
    RandomJitter, GenerateTrimap


def get_train_transforms(resolution: int):
    return transforms.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(resolution, resolution), vertical_bias_ratio=0.2),
        RandomJitter(),
        GenerateTrimap(),  # This generates the trimap from the ground truth alpha.
        ToTensor(),
        Normalize()
    ])


def get_val_transforms(resolution: int):
    return transforms.Compose([
        OriginScale(resolution),
        ToTensor(),
        Normalize()
    ])


def get_test_transforms(resolution: int):
    return transforms.Compose([
        Resize(size=(resolution, resolution)),
        ToTensor(),
        Normalize()
    ])
