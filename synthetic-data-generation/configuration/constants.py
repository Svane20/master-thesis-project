from pydantic import BaseModel


class Constants(BaseModel):
    """
    Constants for the scene generation.

    Attributes:
        playground_directory (str): The directory where the images, masks and blender files are saved to
        save_blend_files (bool): Whether to save the blend files to disk.
        render_images (bool): Whether to render the images in the scene.
        num_iterations (int): The number of images to generate.
        seed (int): The seed for random number generation.
    """
    playground_directory: str
    save_blend_files: bool
    render_images: bool
    num_iterations: int
    seed: int | None
