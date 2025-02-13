from pydantic import BaseModel


class Directories(BaseModel):
    """
    Directories for the project.
    """
    playground_directory: str
    models_directory: str