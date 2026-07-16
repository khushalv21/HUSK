import os
import yaml
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.husk/config.yaml")

DEFAULT_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_key": "",
    "api_url": "http://localhost:11434" # Used as Ollama host URL
}

class ConfigManager:
    """
    Manages local user configurations stored in `~/.husk/config.yaml`.
    """
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        """
        Loads YAML configuration from disk.
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded = yaml.safe_load(f)
                    if loaded and isinstance(loaded, dict):
                        self.config.update(loaded)
            except Exception:
                pass

    def save(self):
        """
        Saves YAML configuration to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
        except Exception:
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save()
