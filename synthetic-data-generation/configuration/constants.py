from pydantic import BaseModel


class Constants(BaseModel):
    """
    Constants for the scene generation.

    Attributes:
        project_name (str): The name of the project.
        seed (int): The seed for random number generation.
    """
    project_name: str
    save_blend_files: bool
    render_images: bool
    seed: int | None