"""
CVD (Cumulative Volume Delta) Engine
Tracks buy vs sell volume pressure
"""
from typing import Optional
from collections import deque
from datetime import datetime, timezone

from ...utils.logger import get_logger


logger = get_logger(__name__)


class CVDEngine:
    """
    Cumulative Volume Delta calculator.
    
    CVD = Cumulative(Buy Volume - Sell Volume)
    
    Positive CVD = More buying pressure
    Negative CVD = More selling pressure
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of trades to track
        """
        self.window_size = window_size
        self.trades = deque(maxlen=window_size)
        
        self.cumulative_buy_volume = 0.0
        self.cumulative_sell_volume = 0.0
        self.cvd = 0.0
        
        # For detecting divergences
        self.price_high = 0.0
        self.price_low = float('inf')
        self.cvd_at_price_high = 0.0
        self.cvd_at_price_low = 0.0
    
    def add_trade(
        self,
        price: float,
        volume: float,
        side: Optional[str] = None,
        is_buyer_maker: Optional[bool] = None
    ) -> float:
        """
        Add a trade and update CVD.
        
        Args:
            price: Trade price
            volume: Trade volume
            side: 'buy' or 'sell' (if known)
            is_buyer_maker: True if buyer is maker (sell pressure)
            
        Returns:
            Updated CVD
        """
        # Determine if buy or sell
        if side:
            is_buy = (side.lower() == 'buy')
        elif is_buyer_maker is not None:
            is_buy = not is_buyer_maker  # If buyer is maker, it's a sell
        else:
            # Fallback: assume buy if we don't know
            is_buy = True
        
        # Remove oldest if window full
        if len(self.trades) == self.window_size:
            old_trade = self.trades[0]
            if old_trade['is_buy']:
                self.cumulative_buy_volume -= old_trade['volume']
            else:
                self.cumulative_sell_volume -= old_trade['volume']
        
        # Add new trade
        self.trades.append({
            'price': price,
            'volume': volume,
            'is_buy': is_buy,
            'timestamp': datetime.now(timezone.utc)
        })
        
        if is_buy:
            self.cumulative_buy_volume += volume
        else:
            self.cumulative_sell_volume += volume
        
        # Calculate CVD
        self.cvd = self.cumulative_buy_volume - self.cumulative_sell_volume
        
        # Track price extremes for divergence detection
        if price > self.price_high:
            self.price_high = price
            self.cvd_at_price_high = self.cvd
        
        if price < self.price_low:
            self.price_low = price
            self.cvd_at_price_low = self.cvd
        
        return self.cvd
    
    def get_cvd(self) -> float:
        """Get current CVD"""
        return self.cvd
    
    def get_cvd_trend(self) -> str:
        """
        Get CVD trend.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if self.cvd > 0:
            return 'bullish'
        elif self.cvd < 0:
            return 'bearish'
        else:
            return 'neutral'
    
    def detect_divergence(self, current_price: float) -> Optional[str]:
        """
        Detect bullish/bearish divergence.
        
        Bullish divergence: Price makes lower low, CVD makes higher low
        Bearish divergence: Price makes higher high, CVD makes lower high
        
        Returns:
            'bullish_divergence', 'bearish_divergence', or None
        """
        # Bearish divergence
        if (current_price > self.price_high * 0.99 and 
            self.cvd < self.cvd_at_price_high * 0.95):
            return 'bearish_divergence'
        
        # Bullish divergence
        if (current_price < self.price_low * 1.01 and 
            self.cvd > self.cvd_at_price_low * 1.05):
            return 'bullish_divergence'
        
        return None
    
    def get_stats(self) -> dict:
        """Get CVD statistics"""
        return {
            'cvd': self.cvd,
            'cvd_trend': self.get_cvd_trend(),
            'total_buy_volume': self.cumulative_buy_volume,
            'total_sell_volume': self.cumulative_sell_volume,
            'buy_sell_ratio': (
                self.cumulative_buy_volume / self.cumulative_sell_volume
                if self.cumulative_sell_volume > 0
                else 0
            ),
            'trade_count': len(self.trades),
        }
    
    def reset(self):
        """Reset CVD calculator"""
        self.trades.clear()
        self.cumulative_buy_volume = 0.0
        self.cumulative_sell_volume = 0.0
        self.cvd = 0.0
        self.price_high = 0.0
        self.price_low = float('inf')