from pathlib import Path
import asyncio
import gevent
import csv
from asgi_lifespan import LifespanManager
from locust.log import setup_logging
from locust.env import Environment
from locust.stats import StatsCSV, stats_printer, stats_history

from tests.utils.performance import serve_app_in_background, APIUser

# Project and Model Type
current_directory = Path(__file__).parent
project_name = current_directory.parent.name
model_type = current_directory.name


def run_locust_headless(
        project_name: str,
        model_type: str,
        use_gpu: bool = False,
        users: int = 50,
        spawn_rate: int = 5,
        run_time_s: int = 10,
        csv_prefix: str = "./reports/locust_report",
):
    # 1) Launch ASGI app + test client
    app, client, transport = asyncio.run(
        serve_app_in_background(project_name, model_type, use_gpu=use_gpu)
    )

    # 2) Configure Locust
    base_url = str(client.base_url)
    APIUser.host = base_url
    env = Environment(user_classes=[APIUser], host=base_url)

    # 3) Ensure CSV output directory exists
    output_dir = Path(csv_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4) Setup Locust logging & live stats printers
    setup_logging("INFO")
    gevent.spawn(stats_printer, env.stats)  # OK: env.stats is initialized

    # 5) Start the test
    runner = env.create_local_runner()
    gevent.spawn(stats_history, runner)  # FIXED: use the real runner, not env.runner

    runner.start(user_count=users, spawn_rate=spawn_rate)
    print(f"🔹 Running {users} users @ {spawn_rate}/s for {run_time_s}s against {base_url}")
    gevent.spawn_later(run_time_s, runner.quit)
    runner.greenlet.join()

    # 6) MANUALLY DUMP CSVs
    percentiles = [0.50, 0.95, 0.99]
    stats_csv = StatsCSV(environment=env, percentiles_to_report=percentiles)

    with open(f"{csv_prefix}_stats.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats_csv.requests_csv_columns)
        stats_csv._requests_data_rows(writer)

    with open(f"{csv_prefix}_failures.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats_csv.failures_columns)
        stats_csv._failures_data_rows(writer)

    with open(f"{csv_prefix}_exceptions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(stats_csv.exceptions_columns)
        stats_csv._exceptions_data_rows(writer)

    # 7) Cleanly shut down ASGI
    async def _shutdown_asgi():
        await client.aclose()
        await transport.aclose()
        try:
            await LifespanManager(app).__aexit__(None, None, None)
        except TimeoutError:
            pass

    asyncio.run(_shutdown_asgi())


if __name__ == "__main__":
    run_locust_headless(project_name, model_type)
