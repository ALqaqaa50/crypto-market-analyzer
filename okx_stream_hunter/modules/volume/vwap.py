"""
VWAP (Volume Weighted Average Price) Calculator
"""
from typing import Optional
from collections import deque

from ...utils.logger import get_logger


logger = get_logger(__name__)


class VWAPCalculator:
    """
    Calculate Volume Weighted Average Price.
    
    VWAP = Sum(Price * Volume) / Sum(Volume)
    
    Used to determine average price weighted by volume.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of trades to include in rolling VWAP
        """
        self.window_size = window_size
        self.trades = deque(maxlen=window_size)
        self.cumulative_pv = 0.0  # Price * Volume
        self.cumulative_volume = 0.0
        self.vwap = 0.0
    
    def add_trade(self, price: float, volume: float) -> float:
        """
        Add a trade and recalculate VWAP.
        
        Args:
            price: Trade price
            volume: Trade volume
            
        Returns:
            Updated VWAP
        """
        pv = price * volume
        
        # If window is full, remove oldest trade
        if len(self.trades) == self.window_size:
            old_trade = self.trades[0]
            self.cumulative_pv -= old_trade['pv']
            self.cumulative_volume -= old_trade['volume']
        
        # Add new trade
        self.trades.append({'price': price, 'volume': volume, 'pv': pv})
        self.cumulative_pv += pv
        self.cumulative_volume += volume
        
        # Calculate VWAP
        if self.cumulative_volume > 0:
            self.vwap = self.cumulative_pv / self.cumulative_volume
        
        return self.vwap
    
    def get_vwap(self) -> float:
        """Get current VWAP value"""
        return self.vwap
    
    def get_deviation_from_vwap(self, current_price: float) -> float:
        """
        Get price deviation from VWAP in percentage.
        
        Positive = price above VWAP (bearish signal)
        Negative = price below VWAP (bullish signal)
        """
        if self.vwap == 0:
            return 0.0
        return ((current_price - self.vwap) / self.vwap) * 100
    
    def reset(self):
        """Reset calculator"""
        self.trades.clear()
        self.cumulative_pv = 0.0
        self.cumulative_volume = 0.0
        self.vwap = 0.0