"""
Configuration loader with YAML + Environment variables support
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML + environment variables"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to settings.yaml (auto-detected if None)
        """
        if config_path is None:
            # Auto-detect config file
            possible_paths = [
                Path("config/settings.yaml"),
                Path("../config/settings.yaml"),
                Path("/opt/okx_stream_hunter/config/settings.yaml"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    "settings.yaml not found. Please create config/settings.yaml"
                )
        
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
        
        # Override with environment variables
        self._apply_env_overrides()
        
        # Validate
        self.validate()
    
    def _apply_env_overrides(self):
        """Override config values with environment variables"""
        
        # Database URL
        if os.getenv("NEON_DATABASE_URL"):
            self.config.setdefault("database", {})
            self.config["database"]["url"] = os.getenv("NEON_DATABASE_URL")
        
        # Webhooks
        if os.getenv("WEBHOOK_URL"):
            self.config.setdefault("webhooks", {})
            self.config["webhooks"]["snapshot_url"] = os.getenv("WEBHOOK_URL")
        
        if os.getenv("WEBHOOK_HEARTBEAT_URL"):
            self.config.setdefault("webhooks", {})
            self.config["webhooks"]["heartbeat_url"] = os.getenv("WEBHOOK_HEARTBEAT_URL")
        
        # Trading pair
        if os.getenv("TRADING_PAIR"):
            self.config.setdefault("trading", {})
            self.config["trading"]["symbol"] = os.getenv("TRADING_PAIR")
        
        # Log level
        if os.getenv("LOG_LEVEL"):
            self.config.setdefault("logging", {})
            self.config["logging"]["level"] = os.getenv("LOG_LEVEL")
    
    def validate(self):
        """Validate required configuration"""
        required = [
            ("trading", "symbol"),
            ("websocket", "url"),
        ]
        
        for keys in required:
            current = self.config
            for key in keys:
                if key not in current:
                    raise ValueError(f"Missing required config: {'.'.join(keys)}")
                current = current[key]
        
        logger.info("Configuration validated successfully")
    
    def get(self, *keys, default=None):
        """
        Get configuration value by nested keys.
        
        Example:
            config.get("database", "pool_max_size")
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def __getitem__(self, key):
        """Allow dict-like access"""
        return self.config[key]
    
    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self.config


# Global config instance
_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get global configuration instance (singleton)"""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config():
    """Reload configuration from file"""
    global _config
    _config = None
    return get_config()