import torchvision.transforms as transforms

from ...datasets.transforms import Resize, ToTensor, Normalize


def get_test_transforms(resolution: int):
    return transforms.Compose([
        Resize(size=(resolution, resolution)),
        ToTensor(),
        Normalize()
    ])
