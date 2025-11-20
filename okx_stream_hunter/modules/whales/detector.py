"""
Whale Detection Engine
Detects large trades and abnormal market activity
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional
from collections import deque

from ...utils.logger import get_logger
from ...config.loader import get_config


logger = get_logger(__name__)


@dataclass
class WhaleEvent:
    """Whale activity event"""
    timestamp: datetime
    event_type: str  # 'large_trade', 'volume_spike', 'orderbook_whale'
    side: Optional[str]  # 'buy', 'sell'
    size: float
    price: float
    usd_value: float
    magnitude: float  # How many times larger than average
    details: Dict
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class WhaleDetector:
    """
    Detect whale activity in the market.
    
    Features:
    - Large trade detection (USD threshold)
    - Volume spike detection (vs. rolling average)
    - Abnormal activity patterns
    - Multiple detection methods
    """
    
    def __init__(self):
        config = get_config()
        
        # Thresholds
        self.min_usd_threshold = config.get(
            "whale_detection", "min_usd", default=100000
        )  # $100k default
        
        self.volume_multiplier = config.get(
            "whale_detection", "volume_multiplier", default=5.0
        )  # 5x average
        
        self.window_size = config.get(
            "whale_detection", "window_size", default=100
        )
        
        # State
        self.recent_trades: deque = deque(maxlen=self.window_size)
        self.recent_volumes: deque = deque(maxlen=self.window_size)
        
        self.avg_trade_size = 0.0
        self.avg_trade_usd = 0.0
        self.avg_volume = 0.0
        
        # Statistics
        self.total_whales_detected = 0
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.largest_whale_usd = 0.0
        
        logger.info(
            f"Whale Detector initialized: "
            f"USD threshold=${self.min_usd_threshold:,.0f}, "
            f"Volume multiplier={self.volume_multiplier}x"
        )
    
    def process_trade(
        self,
        price: float,
        size: float,
        side: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[WhaleEvent]:
        """
        Process a trade and detect if it's whale activity.
        
        Args:
            price: Trade price
            size: Trade size (BTC)
            side: 'buy' or 'sell' (optional)
            timestamp: Trade timestamp
            
        Returns:
            WhaleEvent if detected, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        usd_value = price * size
        
        # Update rolling averages
        self.recent_trades.append({
            'size': size,
            'usd': usd_value,
            'timestamp': timestamp
        })
        self.recent_volumes.append(size)
        
        if len(self.recent_trades) >= 10:
            self._update_averages()
        
        # Detection logic
        whale_event = None
        
        # Method 1: USD threshold
        if usd_value >= self.min_usd_threshold:
            magnitude = usd_value / self.avg_trade_usd if self.avg_trade_usd > 0 else 0
            whale_event = WhaleEvent(
                timestamp=timestamp,
                event_type='large_trade',
                side=side,
                size=size,
                price=price,
                usd_value=usd_value,
                magnitude=magnitude,
                details={
                    'detection_method': 'usd_threshold',
                    'threshold': self.min_usd_threshold,
                }
            )
        
        # Method 2: Volume spike (multiple of average)
        elif self.avg_volume > 0 and size > (self.avg_volume * self.volume_multiplier):
            magnitude = size / self.avg_volume
            whale_event = WhaleEvent(
                timestamp=timestamp,
                event_type='volume_spike',
                side=side,
                size=size,
                price=price,
                usd_value=usd_value,
                magnitude=magnitude,
                details={
                    'detection_method': 'volume_spike',
                    'multiplier': self.volume_multiplier,
                    'avg_volume': self.avg_volume,
                }
            )
        
        # Record statistics
        if whale_event:
            self.total_whales_detected += 1
            if side == 'buy':
                self.whale_buy_volume += size
            elif side == 'sell':
                self.whale_sell_volume += size
            
            if usd_value > self.largest_whale_usd:
                self.largest_whale_usd = usd_value
            
            logger.warning(
                f"🐋 WHALE DETECTED: {whale_event.event_type.upper()} | "
                f"${usd_value:,.0f} | "
                f"{size:.4f} BTC @ ${price:,.2f} | "
                f"{magnitude:.1f}x average | "
                f"Side: {side or 'unknown'}"
            )
        
        return whale_event
    
    def _update_averages(self):
        """Update rolling averages"""
        if self.recent_trades:
            self.avg_trade_size = sum(t['size'] for t in self.recent_trades) / len(self.recent_trades)
            self.avg_trade_usd = sum(t['usd'] for t in self.recent_trades) / len(self.recent_trades)
        
        if self.recent_volumes:
            self.avg_volume = sum(self.recent_volumes) / len(self.recent_volumes)
    
    def get_whale_pressure(self) -> Dict:
        """
        Get current whale pressure (buy vs sell).
        
        Returns:
            Dict with whale statistics
        """
        net_whale_volume = self.whale_buy_volume - self.whale_sell_volume
        
        return {
            'total_whales': self.total_whales_detected,
            'whale_buy_volume': self.whale_buy_volume,
            'whale_sell_volume': self.whale_sell_volume,
            'net_whale_volume': net_whale_volume,
            'whale_pressure': 'bullish' if net_whale_volume > 0 else 'bearish',
            'largest_whale_usd': self.largest_whale_usd,
            'avg_trade_size': self.avg_trade_size,
            'avg_trade_usd': self.avg_trade_usd,
        }
    
    def reset_stats(self):
        """Reset whale statistics"""
        self.whale_buy_volume = 0.0
        self.whale_sell_volume = 0.0
        self.total_whales_detected = 0
        self.largest_whale_usd = 0.0