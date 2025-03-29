import torchvision.transforms as T

from libs.datasets.synthetic.synthetic_dataset import DatasetPhase
from libs.datasets.transforms import Resize, ToTensor, Normalize, OriginScale, RandomAffine, TopBiasedRandomCrop, \
    RandomJitter, GenerateTrimap, Rescale, GenerateFGBG

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def get_transforms(phase: DatasetPhase, resolution: int = 256, crop_resolution: int = None) -> T.Compose:
    if phase == DatasetPhase.Train:
        return get_train_transforms(resolution, crop_resolution)
    elif phase == DatasetPhase.Val:
        return get_val_transforms(resolution)
    elif phase == DatasetPhase.Test:
        return get_test_transforms(resolution)
    else:
        raise ValueError(f"Invalid phase: {phase}")


def get_train_transforms(resolution: int = 256, crop_resolution: int = None) -> T.Compose:
    crop_size = crop_resolution or resolution

    transforms = [
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(crop_size, crop_size), vertical_bias_ratio=0.4),
        RandomJitter(),
        GenerateTrimap(),
        GenerateFGBG(),
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ]

    if crop_resolution and crop_resolution > resolution:
        transforms.insert(2, OriginScale(output_size=resolution))

    return T.Compose(transforms)


def get_val_transforms(resolution: int = 256) -> T.Compose:
    return T.Compose([
        OriginScale(output_size=resolution),
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ])


def get_test_transforms(resolution: int = 256) -> T.Compose:
    return T.Compose([
        Resize(size=(resolution, resolution)),
        ToTensor(),
        Rescale(scale=0.00392156862745098),
        Normalize(mean=MEAN, std=STD),
    ])
