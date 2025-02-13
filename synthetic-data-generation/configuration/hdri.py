from pydantic import BaseModel
from typing import Dict


class SunConfiguration(BaseModel):
    """
    Configuration for sun settings in the HDRI environment.

    Attributes:
        size (dict): A dictionary containing the 'min' and 'max' size of the sun.
        elevation (dict): A dictionary containing the 'min' and 'max' elevation of the sun in degrees.
        rotation (dict): A dictionary containing the 'min' and 'max' rotation of the sun in degrees.
        intensity (dict): A dictionary containing the 'min' and 'max' intensity of the sun's light.
    """
    size: Dict[str, int]
    elevation: Dict[str, int]
    rotation: Dict[str, int]
    intensity: Dict[str, float]


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
    temperature: Dict[str, int]
    strength: Dict[str, float]
    density: Dict[str, int]
    sky_type: str

    sun_configuration: SunConfiguration
