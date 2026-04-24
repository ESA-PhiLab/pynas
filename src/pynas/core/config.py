from configparser import ConfigParser
from importlib.resources import files
from pathlib import Path


def default_config_path():
    """Return the package-bundled configuration path."""
    return files("pynas.core").joinpath("config.ini")


def load_default_config() -> ConfigParser:
    """Load the package-bundled configuration independent of the working directory."""
    config = ConfigParser()
    with default_config_path().open("r", encoding="utf-8") as config_file:
        config.read_file(config_file)
    return config


def load_config(config_path: str | Path | None = None) -> ConfigParser:
    """Load a user config path or fall back to the package-bundled configuration."""
    if config_path is None:
        return load_default_config()

    config = load_default_config()
    config.read(Path(config_path))
    return config
