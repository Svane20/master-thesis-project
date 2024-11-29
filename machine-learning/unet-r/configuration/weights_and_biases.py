from pydantic import BaseModel, Field


class WeightAndBiasesConfig(BaseModel):
    """
    Configuration class for Weights & Biases initialization.

    Attributes:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for the data loader.
        learning_rate (float): Learning rate for the optimizer.
        learning_rate_decay (float): Learning rate decay for the optimizer.
        seed (int): Random seed for reproducibility.
        dataset (str): Dataset name.
        architecture (str): Model architecture.
        name_of_model (str): Model name for tracking.
        device (str): Device type, e.g., 'cuda' or 'cpu'.
    """
    epochs: int = Field(description="Number of training epochs")
    batch_size: int = Field(description="Batch size for the data loader")
    learning_rate: float = Field(description="Learning rate for the optimizer")
    learning_rate_decay: float = Field(description="Learning rate decay for the optimizer")
    seed: int = Field(description="Random seed for reproducibility")
    dataset: str = Field(description="Dataset name")
    architecture: str = Field(description="Model architecture")
    name_of_model: str = Field(description="Model name for tracking")
    device: str = Field(description="Device type, e.g., 'cuda' or 'cpu'")
