"""
Simple configuration loader for okx_stream_hunter.

- Loads YAML from config/settings.yaml
- Supports ${ENV_VAR} placeholders
- Exposes a tiny `.get(section, key, default)` API
- Allows environment variables to override some values (DB + OKX + Claude)
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.logger import get_logger

logger = get_logger(__name__)

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigLoader:
    """Load and manage project configuration."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        # Try to auto-detect settings.yaml if path not given
        if config_path is None:
            candidates = [
                Path("config/settings.yaml"),
                Path("../config/settings.yaml"),
                Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml",
            ]
            for p in candidates:
                if p.is_file():
                    config_path = str(p)
                    break

        self._path: Optional[Path] = Path(config_path) if config_path else None
        self._data: Dict[str, Any] = {}

        if self._path and self._path.is_file():
            self._load()
        else:
            logger.warning("No config/settings.yaml found – using empty config")
            self._data = {}

        self._apply_env_overrides()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get a value from config.

        Examples:
            db_url = cfg.get("database", "url")
            trading_cfg = cfg.get("trading")
        """
        section_data = self._data.get(section, {})
        if key is None:
            return section_data or default
        return section_data.get(key, default)

    def as_dict(self) -> Dict[str, Any]:
        """Return full config as a dict."""
        return dict(self._data)

    def reload(self) -> None:
        """Reload configuration from disk."""
        if self._path and self._path.is_file():
            self._load()
            self._apply_env_overrides()
        else:
            logger.warning("Cannot reload – config file not found")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            text = self._path.read_text(encoding="utf-8")
            raw = yaml.safe_load(text) or {}
        except Exception as e:
            logger.error("Failed to load config from %s: %s", self._path, e)
            raw = {}

        self._data = self._resolve_placeholders(raw)
        logger.info("Configuration loaded from %s", self._path)

    def _resolve_placeholders(self, value: Any) -> Any:
        """Recursively replace ${VAR} with environment values."""
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(v) for v in value]
        if isinstance(value, str):
            # Entire string is a single placeholder
            m = _ENV_PATTERN.fullmatch(value.strip())
            if m:
                return os.getenv(m.group(1), "")
            # Replace inline occurrences
            return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
        return value

    def _apply_env_overrides(self) -> None:
        """Let some environment variables override values from YAML."""
        # Database URL
        db_url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
        if db_url:
            self._data.setdefault("database", {})["url"] = db_url

        # OKX API credentials
        okx_api_key = os.getenv("OKX_API_KEY")
        okx_secret_key = os.getenv("OKX_SECRET_KEY")
        okx_passphrase = os.getenv("OKX_PASSPHRASE")
        if okx_api_key or okx_secret_key or okx_passphrase:
            okx_cfg = self._data.setdefault("okx", {})
            if okx_api_key:
                okx_cfg["api_key"] = okx_api_key
            if okx_secret_key:
                okx_cfg["secret_key"] = okx_secret_key
            if okx_passphrase:
                okx_cfg["passphrase"] = okx_passphrase

        # Claude API key
        claude_key = os.getenv("CLAUDE_API_KEY")
        if claude_key:
            self._data.setdefault("claude", {})["api_key"] = claude_key


# ----------------------------------------------------------------------
# Module-level singleton helpers
# ----------------------------------------------------------------------
_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Return global ConfigLoader singleton."""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config() -> ConfigLoader:
    """Force reload configuration and return instance."""
    global _config
    _config = ConfigLoader()
    return _config


# ============================
# Compatibility Helper
# ============================
def get_settings() -> ConfigLoader:
    """
    Backward-compatible alias for `get_config()` so older modules keep working.

    Returns the `ConfigLoader` singleton.
    """
    return get_config()
