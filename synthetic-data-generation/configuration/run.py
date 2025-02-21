from enum import Enum
from pydantic import BaseModel


class RunConfiguration(BaseModel):
    """
    Configuration for the number of runs and delay between runs.

    Attributes:
        max_runs (int): The number of times to run the script.
        delay (int): The delay in seconds between runs.
    """
    max_runs: int
    delay: int
