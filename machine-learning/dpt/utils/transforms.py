import torchvision.transforms as transforms

from libs.datasets.transforms import Resize, ToTensor, Normalize, OriginScale, RandomAffine, TopBiasedRandomCrop, \
    RandomJitter, GenerateTrimap, Rescale

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def get_train_transforms(resolution: int = 256) -> transforms.Compose:
    return transforms.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(resolution, resolution), vertical_bias_ratio=0.4),
        RandomJitter(),
        GenerateTrimap(),  # This generates the trimap from the ground truth alpha.
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms(resolution: int = 256) -> transforms.Compose:
    return transforms.Compose([
        OriginScale(output_size=resolution),
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ])


def get_test_transforms(resolution: int = 256) -> transforms.Compose:
    return transforms.Compose([
        Resize(size=(resolution, resolution)),
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ])
