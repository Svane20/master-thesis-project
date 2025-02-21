from pydantic import BaseModel


class RunConfiguration(BaseModel):
    """
    Configuration for the number of runs and delay between runs.

    Attributes:
        max_runs (int): The number of times to run the script.
        delay (int): The delay in seconds between runs.
        log_path (str): The path to the log file.
        save_logs (bool): Whether to save logs to a file.
    """
    max_runs: int
    delay: int
    log_path: str
    save_logs: bool = False
