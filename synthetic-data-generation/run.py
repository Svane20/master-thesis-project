import subprocess
import time
import logging

from custom_logging.custom_logger import setup_logging
from main import get_configuration


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
            else:
                logging.info(f"Error occurred during run {i + 1}: {e}")
        else:
            logging.info(f"Run {i + 1} completed successfully.")

        run_end = time.time()
        run_duration = run_end - run_start
        elapsed_times.append(run_duration)

        # Log the duration of the current run
        minutes, seconds = divmod(run_duration, 60)
        logging.info(f"Run {i + 1} took {int(minutes)} minutes and {seconds:.2f} seconds.")

        # Wait before the next run (except after the final iteration)
        if i < max_runs - 1:
            # Average duration of runs so far (not including the delay)
            average_run_time = sum(elapsed_times) / len(elapsed_times)
            remaining_runs = max_runs - i - 1
            # Estimated remaining time includes both average run time and delay per remaining run
            estimated_remaining_time = (average_run_time + delay) * remaining_runs
            rem_minutes, rem_seconds = divmod(estimated_remaining_time, 60)

            logging.info(
                f"Estimated time remaining: {int(rem_minutes)} minutes and {rem_seconds:.2f} seconds."
            )
            logging.info(f"Waiting {delay} seconds before the next run...\n")

            time.sleep(delay)

    overall_end = time.time()
    total_time = overall_end - overall_start
    total_minutes, total_seconds = divmod(total_time, 60)

    logging.info("All runs are complete.")
    logging.info(f"Total execution time: {int(total_minutes)} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
