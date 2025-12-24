"""Shared configuration loader used by all src modules.

Centralises YAML parsing and environment-variable loading so that
crawlers.py, pipeline.py, and rag_chain.py don't each duplicate this logic.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once at import time so every module gets the variables.
load_dotenv()


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load and return the project configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing all configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
        yaml.YAMLError: If the config file contains invalid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug("Loaded config from '%s'.", config_path)
    return config


def get_env_var(name: str, required: bool = True) -> str | None:
    """Retrieve an environment variable, optionally raising if missing.

    Args:
        name: Name of the environment variable.
        required: If True, raise EnvironmentError when the variable is unset.

    Returns:
        The variable's value, or None if not required and unset.

    Raises:
        EnvironmentError: If required is True and the variable is not set.
    """
    value = os.getenv(name)
    if required and not value:
        raise EnvironmentError(f"{name} is not set. Add it to your .env file.")
    return value
