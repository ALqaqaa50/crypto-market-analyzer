"""
Technical Indicators Calculator
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

try:
    import ta
except ImportError:
    ta = None


@dataclass
class IndicatorSnapshot:
    """Complete indicator snapshot for a timeframe"""
    
    symbol: str
    timeframe: str
    close: float
    rsi_14: float
    ema_20: float
    ema_50: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_middle: float
    bb_upper: float
    bb_lower: float
    stoch_rsi: float
    atr_14: float
    volume: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IndicatorEngine:
    """Maintain rolling candles and compute technical indicators"""
    
    def __init__(self, symbol: str, timeframes: List[str], window: int = 5000):
        self.symbol = symbol
        self.timeframes = timeframes
        self.window = window
        self.frames: Dict[str, pd.DataFrame] = {
            tf: pd.DataFrame(
                columns=[
                    "open_time",
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            )
            for tf in timeframes
        }
    
    def _append_candle(self, timeframe: str, candle_dict: Dict) -> pd.DataFrame:
        """Append candle to dataframe and trim to window size"""
        df = self.frames[timeframe]
        new_row = {
            "open_time": candle_dict["open_time"],
            "close_time": candle_dict["close_time"],
            "open": float(candle_dict["open"]),
            "high": float(candle_dict["high"]),
            "low": float(candle_dict["low"]),
            "close": float(candle_dict["close"]),
            "volume": float(candle_dict["volume"]),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Keep only last N candles
        if len(df) > self.window:
            df = df.iloc[-self.window:].reset_index(drop=True)
        
        self.frames[timeframe] = df
        return df
    
    def update(
        self,
        timeframe: str,
        candle_dict: Dict
    ) -> Tuple[IndicatorSnapshot, Dict]:
        """
        Update indicators with new candle.
        
        Returns:
            (IndicatorSnapshot, dict)
        """
        df = self._append_candle(timeframe, candle_dict)
        
        # Need minimum data for indicators
        if len(df) < 50:
            last_close = float(df["close"].iloc[-1])
            last_vol = float(df["volume"].iloc[-1])
            snap = IndicatorSnapshot(
                symbol=self.symbol,
                timeframe=timeframe,
                close=last_close,
                rsi_14=np.nan,
                ema_20=np.nan,
                ema_50=np.nan,
                macd=np.nan,
                macd_signal=np.nan,
                macd_hist=np.nan,
                bb_middle=np.nan,
                bb_upper=np.nan,
                bb_lower=np.nan,
                stoch_rsi=np.nan,
                atr_14=np.nan,
                volume=last_vol,
            )
            return snap, snap.to_dict()
        
        # Calculate indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        
        if ta is None:
            # Fallback if ta library not available
            snap = IndicatorSnapshot(
                symbol=self.symbol,
                timeframe=timeframe,
                close=float(close.iloc[-1]),
                rsi_14=np.nan,
                ema_20=np.nan,
                ema_50=np.nan,
                macd=np.nan,
                macd_signal=np.nan,
                macd_hist=np.nan,
                bb_middle=np.nan,
                bb_upper=np.nan,
                bb_lower=np.nan,
                stoch_rsi=np.nan,
                atr_14=np.nan,
                volume=float(volume.iloc[-1]),
            )
        else:
            # RSI
            rsi_14 = ta.momentum.RSIIndicator(close, window=14).rsi()
            
            # EMA
            ema_20 = ta.trend.EMAIndicator(close, window=20).ema_indicator()
            ema_50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
            
            # MACD
            macd_indicator = ta.trend.MACD(close)
            macd = macd_indicator.macd()
            macd_signal = macd_indicator.macd_signal()
            macd_hist = macd_indicator.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_middle = bb.bollinger_mavg()
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            
            # ATR
            atr_14 = ta.volatility.AverageTrueRange(
                high, low, close, window=14
            ).average_true_range()
            
            # Stochastic RSI
            stoch = ta.momentum.StochRSIIndicator(
                close, window=14, smooth1=3, smooth2=3
            )
            stoch_rsi = stoch.stochrsi()
            
            snap = IndicatorSnapshot(
                symbol=self.symbol,
                timeframe=timeframe,
                close=float(close.iloc[-1]),
                rsi_14=float(rsi_14.iloc[-1]),
                ema_20=float(ema_20.iloc[-1]),
                ema_50=float(ema_50.iloc[-1]),
                macd=float(macd.iloc[-1]),
                macd_signal=float(macd_signal.iloc[-1]),
                macd_hist=float(macd_hist.iloc[-1]),
                bb_middle=float(bb_middle.iloc[-1]),
                bb_upper=float(bb_upper.iloc[-1]),
                bb_lower=float(bb_lower.iloc[-1]),
                stoch_rsi=float(stoch_rsi.iloc[-1]),
                atr_14=float(atr_14.iloc[-1]),
                volume=float(volume.iloc[-1]),
            )
        
        return snap, snap.to_dict()
