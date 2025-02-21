import subprocess
import time
import logging

from custom_logging.custom_logger import setup_logging

# Configuration
MAX_RUNS = 5  # Number of times to run the script
DELAY = 5  # Delay in seconds between runs


def main():
    # Setup logging
    setup_logging(__name__)

    overall_start = time.time()
    elapsed_times = []

    for i in range(MAX_RUNS):
        run_start = time.time()
        logging.info(f"Starting run {i + 1} of {MAX_RUNS}...")

        try:
            result = subprocess.run(["python", "main.py"], check=True)
        except subprocess.CalledProcessError as e:
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
        if i < MAX_RUNS - 1:
            # Average duration of runs so far (not including the delay)
            average_run_time = sum(elapsed_times) / len(elapsed_times)
            remaining_runs = MAX_RUNS - i - 1
            # Estimated remaining time includes both average run time and delay per remaining run
            estimated_remaining_time = (average_run_time + DELAY) * remaining_runs
            rem_minutes, rem_seconds = divmod(estimated_remaining_time, 60)

            logging.info(
                f"Estimated time remaining: {int(rem_minutes)} minutes and {rem_seconds:.2f} seconds."
            )
            logging.info(f"Waiting {DELAY} seconds before the next run...\n")

            time.sleep(DELAY)

    overall_end = time.time()
    total_time = overall_end - overall_start
    total_minutes, total_seconds = divmod(total_time, 60)

    logging.info("All runs are complete.")
    logging.info(f"Total execution time: {int(total_minutes)} minutes and {total_seconds:.2f} seconds.")


if __name__ == "__main__":
    main()
