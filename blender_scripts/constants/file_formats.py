from enum import Enum


class FileFormat(str, Enum):
    """
    Enumeration for file formats used in rendering output.

    Attributes:
        PNG: PNG image format.
        JPEG: JPEG image format.
        EXR: OpenEXR format for high dynamic range imaging.
    """
    PNG = "PNG"
    JPEG = "JPEG"
    EXR = "EXR"
