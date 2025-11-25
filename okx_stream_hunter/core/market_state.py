"""
Market State - Unified market data structure
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque
from collections import deque
import numpy as np


@dataclass
class MarketState:
    """Unified market state from all streaming sources"""
    
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Price data
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    
    # Volume data
    volume_24h: float = 0.0
    volume_window: float = 0.0
    trade_count: int = 0
    
    # Orderbook data
    orderbook_imbalance: float = 0.5
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    total_depth: float = 0.0
    
    # Trade flow
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    buy_pressure: float = 0.5
    sell_pressure: float = 0.5
    aggressive_buy_ratio: float = 0.0
    aggressive_sell_ratio: float = 0.0
    
    # Technical indicators
    volatility: float = 0.0
    momentum: float = 0.0
    trend_strength: float = 0.0
    
    # Liquidity metrics
    liquidity_depth: float = 1.0
    trade_intensity: float = 0.0
    
    # Raw data buffers (with memory limits to prevent leaks)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=5000))
    orderbook: Optional[Dict] = None
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw data"""
        if self.bid > 0 and self.ask > 0:
            self.spread = self.ask - self.bid
            mid_price = (self.bid + self.ask) / 2
            self.spread_bps = (self.spread / mid_price) * 10000 if mid_price > 0 else 0
            
        total_vol = self.buy_volume + self.sell_volume
        if total_vol > 0:
            self.buy_pressure = self.buy_volume / total_vol
            self.sell_pressure = self.sell_volume / total_vol
        
        if self.orderbook:
            bids = self.orderbook.get('bids', [])
            asks = self.orderbook.get('asks', [])
            
            self.bid_depth = sum(float(b[1]) for b in bids[:10]) if bids else 0
            self.ask_depth = sum(float(a[1]) for a in asks[:10]) if asks else 0
            self.total_depth = self.bid_depth + self.ask_depth
            
            if self.total_depth > 0:
                self.orderbook_imbalance = self.bid_depth / self.total_depth
        
        if len(self.recent_trades) > 10:
            prices = [float(t.get('price', 0)) for t in self.recent_trades[-30:]]
            if prices:
                self.volatility = float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0
                price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                self.momentum = float(price_change)
                self.trend_strength = abs(self.momentum)
        
        if len(self.recent_trades) > 0:
            time_window = 60
            recent = [t for t in self.recent_trades if 
                     (datetime.now() - t.get('timestamp', datetime.now())).total_seconds() < time_window]
            self.trade_intensity = len(recent) / time_window if time_window > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API/logging"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'bid': self.bid,
            'ask': self.ask,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'volume_24h': self.volume_24h,
            'volume_window': self.volume_window,
            'trade_count': self.trade_count,
            'orderbook_imbalance': self.orderbook_imbalance,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'buy_pressure': self.buy_pressure,
            'sell_pressure': self.sell_pressure,
            'volatility': self.volatility,
            'momentum': self.momentum,
            'trend_strength': self.trend_strength,
            'trade_intensity': self.trade_intensity,
        }
