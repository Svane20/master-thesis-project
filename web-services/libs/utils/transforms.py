import torchvision.transforms as T

from typing import Tuple, List


def get_transforms(size: Tuple[int, int], mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
