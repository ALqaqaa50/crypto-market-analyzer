"""
Data validation module - prevents bad data from entering the system
"""
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Set
from collections import deque

from ...utils.logger import get_logger
from ...config.loader import get_config


logger = get_logger(__name__)


class DataValidator:
    """Validate incoming market data for quality and consistency"""
    
    def __init__(self):
        config = get_config()
        
        # Price validation settings
        self.max_deviation = config.get("validation", "max_price_deviation", default=0.05)
        self.min_price = config.get("validation", "min_price", default=1000)
        self.max_price = config.get("validation", "max_price", default=1000000)
        
        # Volume validation
        self.max_volume_mult = config.get("validation", "max_volume_multiplier", default=10)
        
        # Deduplication
        self.dedup_enabled = config.get("validation", "dedup_enabled", default=True)
        self.dedup_window = config.get("validation", "dedup_window_seconds", default=1)
        
        # State
        self.last_price: Optional[float] = None
        self.recent_volumes = deque(maxlen=100)
        self.avg_volume: float = 0.0
        
        # Deduplication cache
        self.recent_ticks: Set[str] = set()
        self.last_cleanup = datetime.now(timezone.utc)
        
        # Statistics
        self.total_ticks = 0
        self.rejected_ticks = 0
        self.rejection_reasons: Dict[str, int] = {}
    
    def validate_trade(
        self,
        price: float,
        size: float,
        timestamp: datetime,
        trade_id: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a trade tick.
        
        Returns:
            (is_valid, rejection_reason)
        """
        self.total_ticks += 1
        
        # 1. Price range check
        if not (self.min_price <= price <= self.max_price):
            self._record_rejection("price_out_of_range")
            return False, f"Price {price} outside valid range"
        
        # 2. Price deviation check
        if self.last_price is not None:
            deviation = abs(price - self.last_price) / self.last_price
            if deviation > self.max_deviation:
                self._record_rejection("price_spike")
                logger.warning(
                    f"Price spike detected: {self.last_price:.2f} → {price:.2f} "
                    f"({deviation*100:.2f}% change)"
                )
                # Don't reject completely, but flag it
                # return False, f"Price deviation {deviation*100:.1f}% too large"
        
        # 3. Volume check
        if self.avg_volume > 0:
            if size > (self.avg_volume * self.max_volume_mult):
                self._record_rejection("volume_spike")
                logger.warning(
                    f"Volume spike: {size:.4f} vs avg {self.avg_volume:.4f}"
                )
                # Don't reject, just log
        
        # 4. Deduplication
        if self.dedup_enabled and trade_id:
            tick_key = f"{trade_id}:{timestamp.timestamp()}"
            if tick_key in self.recent_ticks:
                self._record_rejection("duplicate")
                return False, "Duplicate tick"
            
            self.recent_ticks.add(tick_key)
            
            # Cleanup old entries every minute
            now = datetime.now(timezone.utc)
            if (now - self.last_cleanup).total_seconds() > 60:
                self._cleanup_dedup_cache()
        
        # 5. Timestamp validation
        now = datetime.now(timezone.utc)
        age = (now - timestamp).total_seconds()
        
        if age < -5:  # Future timestamp
            self._record_rejection("future_timestamp")
            return False, "Timestamp in the future"
        
        if age > 300:  # Very old data (5 minutes)
            self._record_rejection("stale_data")
            logger.warning(f"Stale data: {age:.0f} seconds old")
        
        # Update state
        self.last_price = price
        self.recent_volumes.append(size)
        if len(self.recent_volumes) >= 10:
            self.avg_volume = sum(self.recent_volumes) / len(self.recent_volumes)
        
        return True, None
    
    def _record_rejection(self, reason: str):
        """Record a rejection for statistics"""
        self.rejected_ticks += 1
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1
    
    def _cleanup_dedup_cache(self):
        """Clean up old deduplication entries"""
        self.recent_ticks.clear()
        self.last_cleanup = datetime.now(timezone.utc)
    
    def get_stats(self) -> Dict:
        """Get validation statistics"""
        rejection_rate = (
            self.rejected_ticks / self.total_ticks
            if self.total_ticks > 0
            else 0.0
        )
        
        return {
            "total_ticks": self.total_ticks,
            "rejected_ticks": self.rejected_ticks,
            "rejection_rate": rejection_rate,
            "rejection_reasons": self.rejection_reasons,
            "last_price": self.last_price,
            "avg_volume": self.avg_volume,
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.total_ticks = 0
        self.rejected_ticks = 0
        self.rejection_reasons = {}