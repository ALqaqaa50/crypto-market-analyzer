# okx_stream_hunter/modules/strategy/detector.py
"""
🔥 Advanced Strategy Detector - Trend, Range, Breakout, Reversal Detection
"""

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

SignalType = Literal["trend_up", "trend_down", "range", "breakout_up", "breakout_down", "reversal_up", "reversal_down", "unknown"]


@dataclass
class StrategySignal:
    """Strategy detection result"""
    signal_type: SignalType
    confidence: float
    strength: float
    
    # Metrics
    price: float
    support: Optional[float] = None
    resistance: Optional[float] = None
    
    # Trend metrics
    trend_slope: float = 0.0
    trend_strength: float = 0.0
    
    # Volatility
    volatility: float = 0.0
    atr: float = 0.0
    
    # Reason
    reason: str = ""
    metadata: Dict = None


class AdvancedStrategyDetector:
    """
    🔥 Multi-Strategy Pattern Detector
    
    Features:
    - Trend detection (momentum, slope analysis)
    - Range detection (support/resistance identification)
    - Breakout confirmation (volume, volatility)
    - Reversal detection (divergence, exhaustion)
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        trend_threshold: float = 0.015,
        range_threshold: float = 0.005,
        breakout_volume_multiplier: float = 1.5,
    ):
        self.lookback_period = lookback_period
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.breakout_volume_multiplier = breakout_volume_multiplier
        
        # Historical data
        self.price_history = deque(maxlen=lookback_period)
        self.volume_history = deque(maxlen=lookback_period)
        self.high_history = deque(maxlen=lookback_period)
        self.low_history = deque(maxlen=lookback_period)
    
    def update(self, candle: Dict) -> None:
        """Update historical data with new candle"""
        self.price_history.append(float(candle.get("close", 0)))
        self.volume_history.append(float(candle.get("volume", 0)))
        self.high_history.append(float(candle.get("high", 0)))
        self.low_history.append(float(candle.get("low", 0)))
    
    def calculate_trend(self) -> Tuple[float, float]:
        """
        Calculate trend using linear regression
        
        Returns: (slope, r_squared)
        """
        if len(self.price_history) < 20:
            return 0.0, 0.0
        
        prices = list(self.price_history)
        n = len(prices)
        x = list(range(n))
        
        # Linear regression
        sum_x = sum(x)
        sum_y = sum(prices)
        sum_xy = sum(x[i] * prices[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        sum_y2 = sum(yi ** 2 for yi in prices)
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0, 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((prices[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((prices[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return slope, r_squared
    
    def detect_support_resistance(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Identify support and resistance levels
        
        Uses swing highs/lows clustering
        """
        if len(self.high_history) < 20 or len(self.low_history) < 20:
            return None, None
        
        highs = list(self.high_history)
        lows = list(self.low_history)
        
        # Find swing highs (local maxima)
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
        
        # Find swing lows (local minima)
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        # Resistance = recent swing high
        resistance = max(swing_highs[-5:]) if len(swing_highs) >= 5 else (max(highs) if highs else None)
        
        # Support = recent swing low
        support = min(swing_lows[-5:]) if len(swing_lows) >= 5 else (min(lows) if lows else None)
        
        return support, resistance
    
    def detect_trend(self) -> Optional[StrategySignal]:
        """
        Detect trending market
        
        Criteria:
        - Clear slope (>= threshold)
        - High R-squared (strong linear fit)
        - Consistent momentum
        """
        if len(self.price_history) < 30:
            return None
        
        slope, r_squared = self.calculate_trend()
        
        prices = list(self.price_history)
        current_price = prices[-1]
        mean_price = statistics.mean(prices)
        volatility = statistics.pstdev(prices) / mean_price if mean_price > 0 else 0
        
        # Normalize slope
        normalized_slope = slope / mean_price if mean_price > 0 else 0
        
        # Trend strength (combination of slope magnitude and R-squared)
        trend_strength = abs(normalized_slope) * r_squared
        
        # Trending if slope significant and R-squared high
        if abs(normalized_slope) >= self.trend_threshold and r_squared >= 0.6:
            signal_type = "trend_up" if normalized_slope > 0 else "trend_down"
            confidence = min(1.0, trend_strength * 50)  # Scale to 0-1
            
            return StrategySignal(
                signal_type=signal_type,
                confidence=confidence,
                strength=trend_strength,
                price=current_price,
                trend_slope=normalized_slope,
                trend_strength=trend_strength,
                volatility=volatility,
                reason=f"Strong trend detected: slope={normalized_slope:.4f}, R²={r_squared:.3f}",
            )
        
        return None
    
    def detect_range(self) -> Optional[StrategySignal]:
        """
        Detect ranging market
        
        Criteria:
        - Low volatility
        - Price oscillating within narrow band
        - Weak trend (low R-squared or small slope)
        """
        if len(self.price_history) < 30:
            return None
        
        prices = list(self.price_history)
        current_price = prices[-1]
        
        mean_price = statistics.mean(prices)
        price_range = (max(prices) - min(prices)) / mean_price if mean_price > 0 else 0
        volatility = statistics.pstdev(prices) / mean_price if mean_price > 0 else 0
        
        slope, r_squared = self.calculate_trend()
        normalized_slope = slope / mean_price if mean_price > 0 else 0
        
        support, resistance = self.detect_support_resistance()
        
        # Range criteria
        is_range = (
            price_range < self.range_threshold * 3 and  # Tight range
            abs(normalized_slope) < self.trend_threshold / 2 and  # Weak trend
            volatility < 0.01  # Low volatility
        )
        
        if is_range:
            confidence = 1.0 - (price_range / (self.range_threshold * 3))
            confidence = min(1.0, max(0.3, confidence))
            
            return StrategySignal(
                signal_type="range",
                confidence=confidence,
                strength=1.0 - volatility / 0.01,
                price=current_price,
                support=support,
                resistance=resistance,
                volatility=volatility,
                reason=f"Range detected: range={price_range:.4f}, volatility={volatility:.4f}",
            )
        
        return None
    
    def detect_breakout(self) -> Optional[StrategySignal]:
        """
        Detect breakout from range
        
        Criteria:
        - Price breaks support/resistance
        - Volume surge (confirmation)
        - Volatility expansion
        """
        if len(self.price_history) < 30 or len(self.volume_history) < 30:
            return None
        
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        current_price = prices[-1]
        current_volume = volumes[-1]
        
        support, resistance = self.detect_support_resistance()
        
        if not support or not resistance:
            return None
        
        # Check for price break
        broke_resistance = current_price > resistance * 1.002  # 0.2% above
        broke_support = current_price < support * 0.998  # 0.2% below
        
        if not broke_resistance and not broke_support:
            return None
        
        # Volume confirmation
        avg_volume = statistics.mean(volumes[-20:-1]) if len(volumes) > 20 else statistics.mean(volumes)
        volume_surge = current_volume > avg_volume * self.breakout_volume_multiplier
        
        # Recent volatility
        recent_prices = prices[-10:]
        volatility = statistics.pstdev(recent_prices) / statistics.mean(recent_prices) if recent_prices else 0
        
        if broke_resistance:
            signal_type = "breakout_up"
            confidence = 0.6 + (0.3 if volume_surge else 0) + (min(0.1, volatility * 10))
        else:
            signal_type = "breakout_down"
            confidence = 0.6 + (0.3 if volume_surge else 0) + (min(0.1, volatility * 10))
        
        return StrategySignal(
            signal_type=signal_type,
            confidence=min(1.0, confidence),
            strength=current_volume / avg_volume if avg_volume > 0 else 1.0,
            price=current_price,
            support=support,
            resistance=resistance,
            volatility=volatility,
            reason=f"Breakout: price={current_price:.2f}, vol_ratio={current_volume/avg_volume:.2f}",
        )
    
    def detect_reversal(self) -> Optional[StrategySignal]:
        """
        Detect trend reversal
        
        Criteria:
        - Momentum exhaustion (decreasing slope)
        - Price at extreme (near support/resistance)
        - Volume divergence (optional)
        """
        if len(self.price_history) < 50:
            return None
        
        prices = list(self.price_history)
        current_price = prices[-1]
        
        # Recent trend
        recent_slope, _ = self.calculate_trend()
        mean_price = statistics.mean(prices)
        normalized_slope = recent_slope / mean_price if mean_price > 0 else 0
        
        # Historical trend (older data)
        old_prices = prices[:len(prices)//2]
        if len(old_prices) < 20:
            return None
        
        # Calculate old slope
        n = len(old_prices)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(old_prices)
        sum_xy = sum(x[i] * old_prices[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        old_slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        old_mean = statistics.mean(old_prices)
        old_normalized_slope = old_slope / old_mean if old_mean > 0 else 0
        
        # Momentum exhaustion: slope decreasing
        slope_change = normalized_slope - old_normalized_slope
        
        support, resistance = self.detect_support_resistance()
        
        # Reversal conditions
        if old_normalized_slope > self.trend_threshold and slope_change < -self.trend_threshold / 2:
            # Was uptrend, momentum fading
            if resistance and current_price > resistance * 0.995:
                signal_type = "reversal_down"
                confidence = min(1.0, abs(slope_change) / self.trend_threshold)
                
                return StrategySignal(
                    signal_type=signal_type,
                    confidence=confidence,
                    strength=abs(slope_change),
                    price=current_price,
                    support=support,
                    resistance=resistance,
                    trend_slope=normalized_slope,
                    reason=f"Reversal: momentum exhausted at resistance",
                )
        
        elif old_normalized_slope < -self.trend_threshold and slope_change > self.trend_threshold / 2:
            # Was downtrend, momentum fading
            if support and current_price < support * 1.005:
                signal_type = "reversal_up"
                confidence = min(1.0, abs(slope_change) / self.trend_threshold)
                
                return StrategySignal(
                    signal_type=signal_type,
                    confidence=confidence,
                    strength=abs(slope_change),
                    price=current_price,
                    support=support,
                    resistance=resistance,
                    trend_slope=normalized_slope,
                    reason=f"Reversal: momentum exhausted at support",
                )
        
        return None
    
    def detect_all(self) -> List[StrategySignal]:
        """
        Run all detection algorithms
        
        Returns list of detected signals (may be empty)
        """
        if len(self.price_history) < 30:
            return []
        
        signals = []
        
        # Try each detector
        detectors = [
            self.detect_trend,
            self.detect_range,
            self.detect_breakout,
            self.detect_reversal,
        ]
        
        for detector in detectors:
            try:
                signal = detector()
                if signal:
                    signals.append(signal)
            except Exception:
                continue
        
        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        
        return signals
    
    def get_best_signal(self) -> Optional[StrategySignal]:
        """Get highest confidence signal"""
        signals = self.detect_all()
        return signals[0] if signals else None


__all__ = ["AdvancedStrategyDetector", "StrategySignal"]
