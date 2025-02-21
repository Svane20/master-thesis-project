from pydantic import BaseModel


class RunConfiguration(BaseModel):
    """
    Configuration for the number of runs and delay between runs.

    Attributes:
        max_runs (int): The number of times to run the script.
        delay (int): The delay in seconds between runs.
        run_path (str): Where the run logs should be stored.
        app_path (str): Where the app logs should be stored.
        save_logs (bool): Whether to save logs to a file.
    """
    max_runs: int
    delay: int
    run_path: str
    app_path: str
    save_logs: bool = False
