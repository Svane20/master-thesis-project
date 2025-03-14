from enum import Enum


class FileExtension(str, Enum):
    """
    Enumeration for common file extensions used in the project.

    Attributes:
        PNG: Represents a '.png' image file.
        BLEND: Represents a '.blend' Blender scene file.
        EXR: Represents an '.exr' OpenEXR image file.
        HDR: Represents an '.hdr' HDR image file.
    """
    PNG = "png"
    JPG = "jpg"
    BLEND = "blend"
    EXR = "exr"
    HDR = "hdr"
