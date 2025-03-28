import torchvision.transforms as T

from libs.datasets.synthetic.synthetic_dataset import DatasetPhase
from libs.datasets.transforms import Resize, ToTensor, Normalize, OriginScale, RandomAffine, TopBiasedRandomCrop, \
    RandomJitter, GenerateTrimap, Rescale

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(phase: DatasetPhase, resolution: int = 224) -> T.Compose:
    if phase == DatasetPhase.Train:
        return get_train_transforms(resolution)
    elif phase == DatasetPhase.Val:
        return get_val_transforms(resolution)
    elif phase == DatasetPhase.Test:
        return get_test_transforms(resolution)
    else:
        raise ValueError(f"Invalid phase: {phase}")


def get_train_transforms(resolution: int = 224) -> T.Compose:
    return T.Compose([
        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
        TopBiasedRandomCrop(output_size=(323, 323), vertical_bias_ratio=0.4),
        OriginScale(output_size=resolution),
        RandomJitter(),
        GenerateTrimap(),  # This generates the trimap from the ground truth alpha.
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=MEAN, std=STD),
    ])


def get_val_transforms(resolution: int = 224) -> T.Compose:
    return T.Compose([
        OriginScale(output_size=resolution),
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=MEAN, std=STD),
    ])


def get_test_transforms(resolution: int = 224) -> T.Compose:
    return T.Compose([
        Resize(size=(resolution, resolution)),
        ToTensor(),
        Rescale(scale=1 / 255.0),
        Normalize(mean=MEAN, std=STD),
    ])
