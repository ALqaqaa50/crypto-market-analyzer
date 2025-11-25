# okx_stream_hunter/state.py
"""
Shared State Manager for Dashboard
Provides a singleton pattern to share live data between main.py and dashboard API
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import threading


@dataclass
class SystemState:
    """
    Global system state that holds live data from AI Brain and Stream Engine.
    This is updated by main.py and read by FastAPI dashboard endpoints.
    """
    
    # AI Signal data
    ai_signal: Optional[Dict[str, Any]] = None
    ai_confidence: float = 0.0
    ai_direction: str = "flat"
    ai_reason: str = "initializing"
    ai_regime: str = "unknown"
    ai_scores: Dict[str, float] = field(default_factory=dict)
    
    # Price data
    current_price: Optional[float] = None
    price_change_24h: float = 0.0
    
    # Strategy data (Entry/TP/SL)
    entry_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    
    # Orderflow data
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int = 0
    whale_trades: int = 0
    max_trade_size: float = 0.0
    cvd: float = 0.0  # Cumulative Volume Delta
    
    # Orderbook data
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    spread: float = 0.0
    orderbook_imbalance: float = 0.5  # 0=all asks, 1=all bids, 0.5=balanced
    
    # Spoof detection
    spoof_risk: float = 0.0
    
    # Position data
    position_direction: str = "flat"
    position_size: float = 0.0
    position_entry_price: Optional[float] = None
    position_pnl: float = 0.0
    
    # System status
    ai_enabled: bool = True
    auto_trading_enabled: bool = False
    last_update: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # 🛡️ Safety system tracking (NEW)
    total_rejections: int = 0
    last_rejection_reason: Optional[str] = None
    emergency_stop_active: bool = False
    
    # 🐋 Whale Detection Tracking (NEW)
    whale_events: list = field(default_factory=list)
    whale_count: int = 0
    last_whale_event: Optional[Dict[str, Any]] = None
    
    # 📊 CVD & Volume Metrics (NEW)
    cvd_value: float = 0.0
    cvd_trend: str = "neutral"  # bullish/bearish/neutral
    
    # 🕯️ Candles Data (NEW)
    candles_1m: list = field(default_factory=list)
    candles_5m: list = field(default_factory=list)
    candles_15m: list = field(default_factory=list)
    candles_1h: list = field(default_factory=list)
    last_candle_closed: Optional[datetime] = None
    
    # Additional signal fields
    signal_direction: str = "flat"
    signal_confidence: float = 0.0
    signal_reason: str = ""
    last_price: Optional[float] = None
    regime: str = "unknown"
    
    # Timestamps
    signal_timestamp: Optional[datetime] = None
    ticker_timestamp: Optional[datetime] = None
    orderbook_timestamp: Optional[datetime] = None
    
    # Lock for thread-safe updates
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    
    def update_from_signal(self, signal: Optional[Dict[str, Any]]) -> None:
        """Update state from AI Brain signal"""
        if not signal:
            return
            
        with self._lock:
            self.ai_signal = signal
            self.ai_confidence = signal.get("confidence", 0.0)
            self.ai_direction = signal.get("direction", "flat")
            self.ai_reason = signal.get("reason", "")
            self.ai_regime = signal.get("regime", "unknown")
            self.ai_scores = signal.get("scores", {})
            self.current_price = signal.get("price")
            
            # Update convenience fields
            self.signal_direction = self.ai_direction
            self.signal_confidence = self.ai_confidence
            self.signal_reason = self.ai_reason
            self.last_price = self.current_price
            self.regime = self.ai_regime
            
            self.signal_timestamp = datetime.utcnow()
            self.last_update = datetime.utcnow()
    
    def update_from_ticker(self, price: float) -> None:
        """Update state from ticker data"""
        with self._lock:
            self.current_price = price
            self.ticker_timestamp = datetime.utcnow()
            self.last_update = datetime.utcnow()
    
    def update_from_orderbook(
        self, 
        bid: Optional[float], 
        ask: Optional[float],
        bid_vol: float = 0.0,
        ask_vol: float = 0.0
    ) -> None:
        """Update state from orderbook data"""
        with self._lock:
            self.best_bid = bid
            self.best_ask = ask
            self.bid_volume = bid_vol
            self.ask_volume = ask_vol
            
            if bid and ask:
                self.spread = (ask - bid) / bid if bid > 0 else 0.0
            
            # Calculate orderbook imbalance (0-1 scale)
            total_vol = bid_vol + ask_vol
            if total_vol > 0:
                self.orderbook_imbalance = bid_vol / total_vol
            
            self.orderbook_timestamp = datetime.utcnow()
            self.last_update = datetime.utcnow()
    
    def update_from_trades(
        self, 
        buy_vol: float, 
        sell_vol: float,
        trade_count: int = 0,
        whale_trades: int = 0,
        max_size: float = 0.0,
        cvd: float = 0.0
    ) -> None:
        """Update state from trade flow data"""
        with self._lock:
            self.buy_volume = buy_vol
            self.sell_volume = sell_vol
            self.trade_count = trade_count
            self.whale_trades = whale_trades
            self.max_trade_size = max_size
            self.cvd = cvd  # Cumulative Volume Delta
            self.last_update = datetime.utcnow()
    
    def update_whale_events(self, whale_events: list, whale_count: int) -> None:
        """Update whale detection data"""
        with self._lock:
            self.whale_events = whale_events[-50:]  # Keep last 50
            self.whale_count = whale_count
            if whale_events:
                self.last_whale_event = whale_events[-1].__dict__ if hasattr(whale_events[-1], '__dict__') else whale_events[-1]
            self.last_update = datetime.utcnow()
    
    def update_cvd_metrics(self, cvd_value: float, cvd_trend: str = "neutral") -> None:
        """Update CVD metrics"""
        with self._lock:
            self.cvd_value = cvd_value
            self.cvd_trend = cvd_trend
            self.last_update = datetime.utcnow()
    
    def update_candles(
        self, 
        candles_1m: list = None,
        candles_5m: list = None,
        candles_15m: list = None,
        candles_1h: list = None
    ) -> None:
        """Update candles data"""
        with self._lock:
            if candles_1m is not None:
                self.candles_1m = [c.to_dict() if hasattr(c, 'to_dict') else c for c in candles_1m[-100:]]
            if candles_5m is not None:
                self.candles_5m = [c.to_dict() if hasattr(c, 'to_dict') else c for c in candles_5m[-100:]]
            if candles_15m is not None:
                self.candles_15m = [c.to_dict() if hasattr(c, 'to_dict') else c for c in candles_15m[-100:]]
            if candles_1h is not None:
                self.candles_1h = [c.to_dict() if hasattr(c, 'to_dict') else c for c in candles_1h[-100:]]
            self.last_candle_closed = datetime.utcnow()
            self.last_update = datetime.utcnow()
    
    def update_strategy(
        self,
        entry: Optional[float],
        tp: Optional[float],
        sl: Optional[float]
    ) -> None:
        """Update strategy levels (Entry/TP/SL)"""
        with self._lock:
            self.entry_price = entry
            self.take_profit = tp
            self.stop_loss = sl
            self.last_update = datetime.utcnow()
    
    def update_position(
        self,
        direction: str = "flat",
        size: float = 0.0,
        entry: Optional[float] = None,
        pnl: float = 0.0
    ) -> None:
        """Update current position data"""
        with self._lock:
            self.position_direction = direction
            self.position_size = size
            self.position_entry_price = entry
            self.position_pnl = pnl
            self.last_update = datetime.utcnow()
    
    def update_system_status(
        self,
        ai_enabled: bool = True,
        auto_trading: bool = False,
        uptime: float = 0.0
    ) -> None:
        """Update system status flags"""
        with self._lock:
            self.ai_enabled = ai_enabled
            self.auto_trading_enabled = auto_trading
            self.uptime_seconds = uptime
            self.last_heartbeat = datetime.utcnow()
            self.last_update = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for API responses"""
        with self._lock:
            return {
                "signal": self.ai_direction,
                "confidence": self.ai_confidence,
                "direction": self.ai_direction,
                "reason": self.ai_reason,
                "regime": self.ai_regime,
                "price": self.current_price,
                "scores": self.ai_scores,
                "buy_volume": self.buy_volume,
                "sell_volume": self.sell_volume,
                "trade_count": self.trade_count,
                "whale_trades": self.whale_trades,
                "max_trade_size": self.max_trade_size,
                "best_bid": self.best_bid,
                "best_ask": self.best_ask,
                "bid_volume": self.bid_volume,
                "ask_volume": self.ask_volume,
                "spread": self.spread,
                "orderbook_imbalance": self.orderbook_imbalance,
                "spoof_risk": self.spoof_risk,
                "entry_price": self.entry_price,
                "take_profit": self.take_profit,
                "stop_loss": self.stop_loss,
                "position": {
                    "direction": self.position_direction,
                    "size": self.position_size,
                    "entry_price": self.position_entry_price,
                    "pnl": self.position_pnl,
                },
                "ai_enabled": self.ai_enabled,
                "auto_trading_enabled": self.auto_trading_enabled,
                "last_update": self.last_update.isoformat() if self.last_update else None,
                "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            }


# Global singleton instance
_system_state: Optional[SystemState] = None
_state_lock = threading.Lock()


def get_system_state() -> SystemState:
    """Get or create the global system state singleton"""
    global _system_state
    
    if _system_state is None:
        with _state_lock:
            if _system_state is None:
                _system_state = SystemState()
    
    return _system_state
