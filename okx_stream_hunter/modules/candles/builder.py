"""
Multi-Timeframe Candle Builder - Build OHLCV candles from trade ticks
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable, List, Optional


@dataclass
class Candle:
    """OHLCV Candle structure"""
    
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int = 0
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "open_time": self.open_time.isoformat() if isinstance(self.open_time, datetime) else self.open_time,
            "close_time": self.close_time.isoformat() if isinstance(self.close_time, datetime) else self.close_time,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "trades": self.trades,
        }


def _timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string to seconds"""
    tf = tf.lower()
    if tf.endswith("s"):
        return int(tf[:-1])
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")


class MultiTimeframeCandleBuilder:
    """Build OHLCV candles for multiple timeframes from trade ticks"""
    
    def __init__(self, symbol: str, timeframes: Iterable[str]):
        self.symbol = symbol
        self.timeframes = list(timeframes)
        self._tf_seconds: Dict[str, int] = {
            tf: _timeframe_to_seconds(tf) for tf in self.timeframes
        }
        self._current: Dict[str, Candle] = {}
    
    @staticmethod
    def _floor_ts(dt: datetime, tf_seconds: int) -> datetime:
        """Floor timestamp to timeframe bucket"""
        ts = int(dt.timestamp())
        bucket = (ts // tf_seconds) * tf_seconds
        return datetime.fromtimestamp(bucket, tz=timezone.utc)
    
    def process_tick(
        self,
        price: float,
        size: float,
        ts: Optional[datetime] = None,
    ) -> List[Candle]:
        """
        Process a trade tick and return any closed candles.
        
        Args:
            price: Trade price
            size: Trade size
            ts: Trade timestamp (defaults to now)
            
        Returns:
            List of closed candles
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        
        closed: List[Candle] = []
        
        for tf, tf_sec in self._tf_seconds.items():
            bucket_open = self._floor_ts(ts, tf_sec)
            bucket_close = bucket_open + timedelta(seconds=tf_sec)
            
            current = self._current.get(tf)
            
            if current is None:
                # Start new candle
                self._current[tf] = Candle(
                    symbol=self.symbol,
                    timeframe=tf,
                    open_time=bucket_open,
                    close_time=bucket_close,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    trades=1,
                )
                continue
            
            if ts >= current.close_time:
                # Close current candle
                closed.append(current)
                
                # Start new candle
                self._current[tf] = Candle(
                    symbol=self.symbol,
                    timeframe=tf,
                    open_time=bucket_open,
                    close_time=bucket_close,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=size,
                    trades=1,
                )
            else:
                # Update current candle
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
                current.volume += size
                current.trades += 1
        
        return closed
    
    def get_current_candles(self) -> Dict[str, Candle]:
        """Get all current (incomplete) candles"""
        return self._current.copy()
