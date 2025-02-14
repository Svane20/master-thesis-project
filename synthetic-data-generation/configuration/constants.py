from pydantic import BaseModel


class Constants(BaseModel):
    """
    Constants for the scene generation.

    Attributes:
        save_blend_files (bool): Whether to save the blend files to disk.
        render_images (bool): Whether to render the images in the scene.
        num_iterations (int): The number of images to generate.
        seed (int): The seed for random number generation.
    """
    save_blend_files: bool
    render_images: bool
    num_iterations: int
    seed: int | None