import pymatting
import numpy as np


def get_foreground_estimation(image, alpha):
    normalized_image = np.array(image) / 255.0
    inverted_alpha = 1 - alpha

    return pymatting.estimate_foreground_ml(image=normalized_image, alpha=inverted_alpha)
