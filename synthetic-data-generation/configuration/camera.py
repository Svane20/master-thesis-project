from pydantic import BaseModel

from constants.defaults import ImageDefaults


class CameraConfiguration(BaseModel):
    """
    Configuration class for the camera settings used in rendering.

    Attributes:
        image_width (int): The width of the rendered image.
        image_height (int): The height of the rendered image.
        camera_fov_mu_deg (float): The mean field of view (FOV) in degrees.
        camera_fov_std_deg (float): The standard deviation of the FOV in degrees (for randomization).
    """
    image_width: int = ImageDefaults.WIDTH
    image_height: int = ImageDefaults.HEIGHT
    camera_fov_mu_deg: float = 60.0
    camera_fov_std_deg: float = 10.0
