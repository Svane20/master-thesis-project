from enum import Enum


class ColorMode(str, Enum):
    """
    Enumeration for different color modes used in rendering output.

    Attributes:
        RGBA: Full color mode with alpha transparency.
        BW: Black and white (grayscale) mode.
    """
    RGBA = "RGBA"
    BW = "BW"
