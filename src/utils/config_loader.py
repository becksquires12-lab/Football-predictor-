"""Load configuration from YAML file with environment variable overrides."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path=None):
    """Load settings from YAML config, with env vars taking precedence for secrets."""
    load_dotenv()

    if config_path is None:
        # Look for config relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "settings.yaml"
        if not config_path.exists():
            config_path = project_root / "config" / "settings.example.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Environment variables override YAML values for secrets
    env_football_data = os.getenv("FOOTBALL_DATA_API_KEY")
    if env_football_data:
        config.setdefault("api_keys", {})["football_data"] = env_football_data

    env_api_football = os.getenv("API_FOOTBALL_KEY")
    if env_api_football:
        config.setdefault("api_keys", {})["api_football"] = env_api_football

    return config


# Singleton config instance
_config = None


def get_config():
    """Get the global config instance, loading it on first access."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
