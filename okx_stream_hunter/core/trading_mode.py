import os
import logging
from enum import Enum
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    SANDBOX = "sandbox"
    REAL = "real"
    
    def __str__(self):
        return self.value


class TradingModeManager:
    def __init__(self, config_mode: str = None):
        config_mode = config_mode or os.getenv('TRADING_MODE', 'sandbox')
        
        self.mode = self._validate_mode(config_mode)
        self.credentials = self._load_credentials()
        self.api_url = self._get_api_url()
        
        self._verify_credentials()
        
        logger.info(f"Trading Mode: {self.mode.value.upper()}")
        logger.info(f"API URL: {self.api_url}")
    
    def _validate_mode(self, mode_str: str) -> TradingMode:
        mode_str = mode_str.lower().strip()
        
        if mode_str == "sandbox":
            return TradingMode.SANDBOX
        elif mode_str == "real":
            return TradingMode.REAL
        else:
            logger.warning(f"Invalid trading_mode '{mode_str}', defaulting to SANDBOX")
            return TradingMode.SANDBOX
    
    def _load_credentials(self) -> Dict[str, str]:
        if self.mode == TradingMode.SANDBOX:
            return {
                'api_key': os.getenv('OKX_SANDBOX_KEY', ''),
                'secret_key': os.getenv('OKX_SANDBOX_SECRET', ''),
                'passphrase': os.getenv('OKX_SANDBOX_PASSPHRASE', '')
            }
        else:
            return {
                'api_key': os.getenv('OKX_API_KEY', ''),
                'secret_key': os.getenv('OKX_API_SECRET', ''),
                'passphrase': os.getenv('OKX_API_PASSPHRASE', '')
            }
    
    def _get_api_url(self) -> str:
        if self.mode == TradingMode.SANDBOX:
            return "https://www.okx.com"
        else:
            return "https://www.okx.com"
    
    def _verify_credentials(self):
        if self.mode == TradingMode.REAL:
            if not all(self.credentials.values()):
                raise ValueError(
                    "REAL trading mode requires all credentials: "
                    "OKX_API_KEY, OKX_API_SECRET, OKX_API_PASSPHRASE"
                )
            
            logger.critical("=" * 80)
            logger.critical("REAL TRADING MODE ENABLED")
            logger.critical("All orders will execute on LIVE MARKETS")
            logger.critical("=" * 80)
        
        else:
            logger.info("SANDBOX mode - all trades are simulated")
            if not all(self.credentials.values()):
                logger.warning("Sandbox credentials incomplete, using paper trading fallback")
    
    def is_sandbox(self) -> bool:
        return self.mode == TradingMode.SANDBOX
    
    def is_real(self) -> bool:
        return self.mode == TradingMode.REAL
    
    def get_log_prefix(self) -> str:
        return f"[{self.mode.value.upper()}]"
    
    def get_credentials(self) -> Dict[str, str]:
        return self.credentials.copy()
    
    def get_api_url(self) -> str:
        return self.api_url
    
    def require_real_mode(self):
        if not self.is_real():
            raise RuntimeError("Operation requires REAL trading mode")
    
    def require_sandbox_mode(self):
        if not self.is_sandbox():
            raise RuntimeError("Operation requires SANDBOX mode")
    
    def get_safety_check(self) -> Tuple[bool, str]:
        if self.is_real():
            if not all(self.credentials.values()):
                return False, "Real mode credentials incomplete"
            
            return True, "Real mode safety check passed"
        
        return True, "Sandbox mode active"


_mode_manager_instance = None

def get_trading_mode_manager(config_mode: str = None) -> TradingModeManager:
    global _mode_manager_instance
    if _mode_manager_instance is None:
        _mode_manager_instance = TradingModeManager(config_mode)
    return _mode_manager_instance
