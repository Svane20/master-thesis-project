import torch

from pathlib import Path
from datetime import datetime
import csv
import logging

from .utils.inference import evaluate_model
from ..configuration.configuration import Config
from ..datasets.synthetic.data_loaders import create_test_data_loader
from ..training.utils.logger import setup_logging

setup_logging(__name__)


def run_evaluation(
        configuration: Config,
        model: torch.nn.Module,
        device: torch.device,
) -> None:
    # Directories
    root_directory = Path(__file__).resolve().parent.parent.parent
    metrics_directory = root_directory / "metrics"

    # Create data loader
    data_loader = create_test_data_loader(config=configuration.dataset, resolution=configuration.scratch.resolution)

    # Model evaluation
    metrics = evaluate_model(model=model, data_loader=data_loader, device=device)

    # Get model name (or fallback)
    model_name = getattr(model, '__class__', type(model)).__name__

    # Get the full checkpoint path
    checkpoint_path = Path(configuration.evaluation.checkpoint_path)
    checkpoint_name = checkpoint_path.stem

    # Prepare CSV row
    timestamp = datetime.now().isoformat()
    num_params = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
    row = {
        "timestamp": timestamp,
        "model": model_name,
        "checkpoint": checkpoint_name,
        "dataset": configuration.dataset.name,
        "resolution": configuration.scratch.resolution,
        "num_parameters": round(num_params, 2),
        "device": device.type,
        "compiled": configuration.evaluation.compile_model,
        **metrics
    }

    # Check if we need to write the header
    csv_path = metrics_directory / "summary.csv"
    write_header = not csv_path.exists()

    # Write or append to CSV
    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    logging.info(f"📊 Appended evaluation to CSV: {csv_path}")
