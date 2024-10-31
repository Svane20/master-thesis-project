from pydantic import BaseModel


class SunConfiguration(BaseModel):
    """
    Configuration for sun settings in the HDRI environment.

    Attributes:
        size (dict): A dictionary containing the 'min' and 'max' size of the sun.
        elevation (dict): A dictionary containing the 'min' and 'max' elevation of the sun in degrees.
        rotation (dict): A dictionary containing the 'min' and 'max' rotation of the sun in degrees.
        intensity (dict): A dictionary containing the 'min' and 'max' intensity of the sun's light.
    """
    size: dict[str, int] = {"min": 1, "max": 3}
    elevation: dict[str, int] = {"min": 45, "max": 90}
    rotation: dict[str, int] = {"min": 0, "max": 360}
    intensity: dict[str, float] = {"min": 0.4, "max": 0.8}


class HDRIConfiguration(BaseModel):
    """
    Configuration for HDRI settings in the scene.

    Attributes:
        temperature (dict): A dictionary containing the 'min' and 'max' temperature values.
        strength (dict): A dictionary containing the 'min' and 'max' strength values for the HDRI.
        density (dict): A dictionary containing the 'min' and 'max' density values for the environment.
        sky_type (str): The type of sky to use in the scene.
        sun_configuration (SunConfiguration): Configuration settings for the sun.
    """
    temperature: dict[str, int] = {"min": 5000, "max": 6500}
    strength: dict[str, float] = {"min": 0.6, "max": 1.0}
    density: dict[str, int] = {"min": 0, "max": 2}
    sky_type: str = "NISHITA"

    sun_configuration: SunConfiguration = SunConfiguration()
