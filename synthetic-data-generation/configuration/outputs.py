from pydantic import BaseModel
from constants.color_mode import ColorMode
from constants.directories import OUTPUT_DIRECTORY
from constants.file_formats import FileFormat


class ImageOutputConfiguration(BaseModel):
    """
    Configuration for image output settings.

    Attributes:
        title (str): The title for the image output.
        use_node_format (bool): Whether to use a custom node format.
        file_format (str): The file format for the output image (e.g., PNG, EXR).
        color_mode (str): The color mode for the image (e.g., RGBA, BW).
        path (str): The output path for the image file.
    """
    title: str = "Image"
    use_node_format: bool = False  # Custom format
    file_format: str = FileFormat.PNG
    color_mode: str = ColorMode.RGBA
    path: str = "Image"


class ObjectIndexOutputConfiguration(BaseModel):
    """
    Configuration for object index output settings.

    Attributes:
        title (str): The title for the object index output.
        use_node_format (bool): Whether to use a custom node format.
        file_format (str): The file format for the output object index (e.g., PNG).
        color_mode (str): The color mode for the output (e.g., BW).
        path (str): The output path for the object index file.
    """
    title: str = "IndexOB"
    use_node_format: bool = False
    file_format: str = FileFormat.PNG
    color_mode: str = ColorMode.BW
    path: str = "IndexOB"


class IDMaskOutputConfiguration(BaseModel):
    """
    Configuration for ID mask output settings.

    Attributes:
        title (str): The title for the ID mask output.
        use_node_format (bool): Whether to use a custom node format.
        file_format (str): The file format for the output ID mask.
        color_mode (str): The color mode for the ID mask.
        path (str): The output path for the ID mask file.
    """
    title: str = "BiomeMask"
    use_node_format: bool = False
    file_format: str = FileFormat.PNG
    color_mode: str = ColorMode.BW
    path: str = "BiomeMask"


class EnvironmentOutputConfiguration(BaseModel):
    """
    Configuration for environment output settings, such as HDRI masks.

    Attributes:
        title (str): The title for the environment output.
        use_node_format (bool): Whether to use a custom node format.
        file_format (str): The file format for the environment output.
        color_mode (str): The color mode for the environment output.
        path (str): The output path for the environment file.
    """
    title: str = "HDRIMask"
    use_node_format: bool = False
    file_format: str = FileFormat.PNG
    color_mode: str = ColorMode.BW
    path: str = "HDRIMask"


class OutputsConfiguration(BaseModel):
    """
    Main configuration class for all output settings.

    Attributes:
        render_image (bool): Whether to render the main image.
        render_object_index (bool): Whether to render the object index.
        render_environment (bool): Whether to render the environment (HDRI).
        output_path (str): The base output path for all rendered files.
    """
    render_image: bool = True
    render_object_index: bool = True
    render_environment: bool = True
    output_path: str = OUTPUT_DIRECTORY.as_posix()

    image_output_configuration: ImageOutputConfiguration = ImageOutputConfiguration()
    object_index_output_configuration: ObjectIndexOutputConfiguration = ObjectIndexOutputConfiguration()
    id_mask_output_configuration: IDMaskOutputConfiguration = IDMaskOutputConfiguration()
    environment_output_configuration: EnvironmentOutputConfiguration = EnvironmentOutputConfiguration()
