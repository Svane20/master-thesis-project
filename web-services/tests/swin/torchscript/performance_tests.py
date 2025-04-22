import time
from pathlib import Path
import subprocess

from tests.utils.performance import setup_api_instance

# Directories
root_directory = Path(__file__).parent.parent.parent
reports_folder = root_directory / "reports"
current_directory = Path(__file__).parent

# Project and Model Type
project_name = current_directory.parent.name
model_type = current_directory.name


def run_locust(users: str, spawn_rate: str, run_time: str, csv_prefix: str) -> None:
    """
    Run locust performance tests.

    Args:
        users (int): Number of concurrent users.
        spawn_rate (int): Rate at which users are spawned.
        run_time (str): Duration for which the test should run.
        csv_prefix (str): Prefix for the CSV report files.
    """
    csv_path = reports_folder / project_name / model_type / csv_prefix

    subprocess.run([
        "locust",
        "-f", str(root_directory / "utils" / "locust.py"),
        "--headless",
        "-u", users,
        "-r", spawn_rate,
        "--run-time", run_time,
        "--host=http://localhost:8000",
        f"--csv={str(csv_path)}",
    ], check=True)


def main():
    # Locust configuration
    users = "10"
    spawn_rate = "2"
    run_time = "2m"

    # API configuration
    workers = 1

    for label, use_gpu in [("gpu", True), ("cpu", False)]:
        print(f"\n=== Starting {label.upper()} run ===")

        # Set up the API instance
        server = setup_api_instance(
            project_name,
            model_type,
            use_gpu,
            workers
        )

        # # Run locust performance tests
        run_locust(
            users,
            spawn_rate,
            run_time,
            csv_prefix=label
        )

        # Stop the API server
        server.should_exit = True

        # Wait for the server to exit
        time.sleep(10)


if __name__ == "__main__":
    main()
