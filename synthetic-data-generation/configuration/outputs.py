from pydantic import BaseModel


class NodeOutputConfiguration(BaseModel):
    """
    Configuration for node output settings.

    Attributes:
        title (str): The title for the image output.
        use_node_format (bool): Whether to use a custom node format.
        file_format (str): The file format for the output image (e.g., PNG, EXR).
        color_mode (str): The color mode for the image (e.g., RGBA, BW).
        path (str): The output path for the image file.
    """
    title: str
    use_node_format: bool
    file_format: str
    color_mode: str
    path: str


class OutputsConfiguration(BaseModel):
    """
    Main configuration class for all output settings.

    Attributes:
        render_image (bool): Whether to render the main image.
        render_object_index (bool): Whether to render the object index.
        render_environment (bool): Whether to render the environment (HDRI).
        output_path (str): The base output path for all rendered files.
    """
    render_image: bool
    render_object_index: bool
    render_environment: bool
    output_path: str

    image_output_configuration: NodeOutputConfiguration
    object_index_output_configuration: NodeOutputConfiguration
    id_mask_output_configuration: NodeOutputConfiguration
    environment_output_configuration: NodeOutputConfiguration
