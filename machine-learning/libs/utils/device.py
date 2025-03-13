import torch

import platform
import logging


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compile_model(model: torch.nn.Module) -> torch.nn.Module:
    backend = "inductor" if platform.system() == "Linux" else "aot_eager"
    logging.info(f"Compiling the model with backend '{backend}'.")

    try:
        model = torch.compile(model, backend=backend)
        logging.info(f"Successfully compiled model with backend '{backend}'.")
    except Exception as e:
        logging.error(f"Model compilation with backend '{backend}' failed: {e}")

    return model
