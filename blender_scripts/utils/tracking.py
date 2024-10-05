import time
import tracemalloc

from custom_logging.custom_logger import setup_logger

logger = setup_logger(__name__)


def start_tracking(project_title: str) -> float:
    """
    Begins tracking the Blender execution.

    Args:
        project_title: The title of the project to track.

    Returns:
        The start time.
    """
    logger.info(f"Started tracking: {project_title}")

    # Start time tracking
    start_time = time.perf_counter()

    # Start memory tracking
    tracemalloc.start()

    return start_time


def end_tracking(project_title: str, start_time: float):
    """
    Ends tracking the Blender execution and logs time and memory stats.

    Args:
        project_title: The title of the project that was tracked.
        start_time: The start time to calculate execution duration.
    """
    # End time tracking
    end_time = time.perf_counter()

    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()

    # Calculate and log time
    elapsed_time = end_time - start_time
    logger.info(f"Finished tracking: {project_title}")
    logger.info(f"Execution time: {elapsed_time:.4f} seconds")

    # Log memory usage (current and peak memory usage)
    logger.info(f"Current memory usage: {current / 10 ** 6:.4f} MB")
    logger.info(f"Peak memory usage: {peak / 10 ** 6:.4f} MB")

    # Stop tracemalloc tracking
    tracemalloc.stop()
