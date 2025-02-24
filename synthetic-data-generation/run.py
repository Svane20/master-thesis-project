import subprocess
import time
import sys
import logging
import os

from custom_logging.custom_logger import setup_logging
from main import get_configuration


def delete_logs_from_previous_runs(log_path: str) -> None:
    """
    Delete the log files from previous runs.

    Args:
        log_path (str): The path to the log file.
    """
    logging.info("Deleting log files from previous runs...")

    log_directory = os.path.dirname(log_path)
    if not os.path.exists(log_directory):
        logging.info("Log directory does not exist. No log files to delete.")
        return

    log_files = [f for f in os.listdir(log_directory) if f.endswith(".log")]
    if not log_files:
        logging.info("No log files found from previous runs.")
        return

    for log_file in log_files:
        os.remove(os.path.join(log_directory, log_file))

    logging.info("Deleted all log files from previous runs.")


def main():
    # Load configuration
    configuration = get_configuration()
    max_runs = configuration.run_configuration.max_runs
    delay = configuration.run_configuration.delay

    # Setup logging
    setup_logging(
        name=__name__,
        log_path=configuration.run_configuration.run_path,
        save_logs=configuration.run_configuration.save_logs
    )

    # Delete log files from previous runs
    delete_logs_from_previous_runs(log_path=configuration.run_configuration.run_path)

    # Time tracking
    overall_start = time.time()
    elapsed_times = []

    for i in range(max_runs):
        run_start = time.time()
        logging.info(f"Starting run {i + 1} of {max_runs}...")

        try:
            result = subprocess.run(["python", "main.py"], check=True)
        except subprocess.CalledProcessError as e:
            # Check if the return code indicates a SIGKILL
            if e.returncode == -9:
                logging.error("Process was killed by SIGKILL. Terminating the script completely.")
                sys.exit(1)  # or break out of the loop if appropriate
            elif e.returncode == -11:
                # Ignore segmentation fault errors
                pass
            else:
                logging.info(f"Error occurred during run {i + 1}: {e}")
        else:
            logging.info(f"Run {i + 1} completed successfully.")

        run_end = time.time()
        current_run_time = run_end - overall_start
        run_duration = run_end - run_start
        elapsed_times.append(run_duration)

        # Log the duration of the current run
        hours, remainder = divmod(run_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        logging.info(f"Run {i + 1} took {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds.")

        # Log the cumulative runtime so far
        run_hours, run_remainder = divmod(current_run_time, 3600)
        run_minutes, run_seconds = divmod(run_remainder, 60)
        logging.info(
            f"Total run time so far: {int(run_hours)} hours, {int(run_minutes)} minutes and {run_seconds:.2f} seconds."
        )

        # Wait before the next run (except after the final iteration)
        if i < max_runs - 1:
            # Average duration of runs so far (not including the delay)
            average_run_time = sum(elapsed_times) / len(elapsed_times)
            remaining_runs = max_runs - i - 1

            # Estimated remaining time includes both average run time and delay per remaining run
            estimated_remaining_time = (average_run_time + delay) * remaining_runs
            est_hours, est_remainder = divmod(estimated_remaining_time, 3600)
            est_minutes, est_seconds = divmod(est_remainder, 60)

            logging.info(
                f"Estimated run time remaining: {int(est_hours)} hours, {int(est_minutes)} minutes and {est_seconds:.2f} seconds."
            )
            logging.info(f"Waiting {delay} seconds before the next run...\n")

            time.sleep(delay)

    overall_end = time.time()
    total_time = overall_end - overall_start
    total_hours, total_remainder = divmod(total_time, 3600)
    total_minutes, total_seconds = divmod(total_remainder, 60)

    logging.info("All runs are complete.")
    logging.info(
        f"Overall execution time: {int(total_hours)} hours, {int(total_minutes)} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
