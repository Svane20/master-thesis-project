from pydantic import BaseModel


class Constants(BaseModel):
    """
    Constants for the scene generation.

    Attributes:
        seed (int): The seed for random number generation.
    """
    save_blend_files: bool
    render_images: bool
    seed: int | None