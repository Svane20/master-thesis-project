from pathlib import Path
from typing import Tuple

from libs.configuration.configuration import Config, ConfigurationMode, get_configuration, \
    get_configuration_and_checkpoint_path, ConfigurationSuffix


def load_config(mode: ConfigurationMode) -> Config:
    return get_configuration(mode=mode, suffix=ConfigurationSuffix.RESNET)


def load_config_and_checkpoint_path(mode: ConfigurationMode) -> Tuple[Config, Path]:
    return get_configuration_and_checkpoint_path(mode=mode, suffix=ConfigurationSuffix.RESNET)
