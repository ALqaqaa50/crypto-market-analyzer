# okx_stream_hunter/modules/tpsl/calculator.py
"""
🔥 Advanced TP/SL Calculator with ATR and Dynamic Volatility Adjustment
"""

import statistics
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass

Direction = Literal["long", "short"]


@dataclass
class TPSLLevels:
    """TP/SL calculation result"""
    tp_price: float
    sl_price: float
    
    # Risk metrics
    risk_amount: float
    reward_amount: float
    rr_ratio: float
    
    # Volatility metrics
    atr: float
    volatility: float
    
    # Method used
    method: str
    confidence: float


class ATRCalculator:
    """Calculate Average True Range for volatility measurement"""
    
    @staticmethod
    def calculate_atr(candles: List[Dict], period: int = 14) -> float:
        """
        Calculate ATR from candle data
        
        Args:
            candles: List of OHLC candles [{"high": ..., "low": ..., "close": ...}, ...]
            period: ATR period (default 14)
        
        Returns:
            ATR value
        """
        if len(candles) < period + 1:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = float(candles[i].get("high", 0))
            low = float(candles[i].get("low", 0))
            prev_close = float(candles[i-1].get("close", 0))
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        # Average True Range
        if len(true_ranges) >= period:
            atr = statistics.mean(true_ranges[-period:])
            return atr
        
        return 0.0


class DynamicTPSLCalculator:
    """
    Advanced TP/SL Calculator
    
    Features:
    - ATR-based dynamic levels
    - Volatility adjustment
    - Microstructure consideration
    - Orderbook imbalance integration
    """
    
    def __init__(
        self,
        min_rr_ratio: float = 1.5,
        default_rr_ratio: float = 2.0,
        atr_multiplier_tp: float = 2.0,
        atr_multiplier_sl: float = 1.0,
    ):
        self.min_rr_ratio = min_rr_ratio
        self.default_rr_ratio = default_rr_ratio
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl
    
    def calculate_volatility(self, prices: List[float]) -> float:
        """Calculate price volatility (standard deviation)"""
        if len(prices) < 10:
            return 0.0
        
        mean = statistics.mean(prices)
        std_dev = statistics.pstdev(prices)
        
        # Normalized volatility
        volatility = std_dev / mean if mean > 0 else 0.0
        return volatility
    
    def calculate_atr_based(
        self,
        entry_price: float,
        direction: Direction,
        candles: List[Dict],
        atr_period: int = 14,
    ) -> TPSLLevels:
        """
        Calculate TP/SL based on ATR
        
        ATR-based levels are adaptive to market volatility
        """
        atr = ATRCalculator.calculate_atr(candles, atr_period)
        
        if atr <= 0:
            # Fallback to percentage-based
            return self.calculate_percentage_based(entry_price, direction, 0.02, 0.01)
        
        # Calculate levels
        if direction == "long":
            sl_price = entry_price - (atr * self.atr_multiplier_sl)
            tp_price = entry_price + (atr * self.atr_multiplier_tp)
        else:  # short
            sl_price = entry_price + (atr * self.atr_multiplier_sl)
            tp_price = entry_price - (atr * self.atr_multiplier_tp)
        
        # Calculate metrics
        risk_amount = abs(entry_price - sl_price)
        reward_amount = abs(tp_price - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Volatility
        prices = [float(c.get("close", 0)) for c in candles[-50:] if c.get("close")]
        volatility = self.calculate_volatility(prices) if prices else 0.0
        
        return TPSLLevels(
            tp_price=tp_price,
            sl_price=sl_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            rr_ratio=rr_ratio,
            atr=atr,
            volatility=volatility,
            method="atr_based",
            confidence=0.85,
        )
    
    def calculate_percentage_based(
        self,
        entry_price: float,
        direction: Direction,
        tp_pct: float = 0.02,
        sl_pct: float = 0.01,
    ) -> TPSLLevels:
        """
        Calculate TP/SL based on fixed percentages
        
        Simple but effective for stable markets
        """
        if direction == "long":
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)
        
        risk_amount = abs(entry_price - sl_price)
        reward_amount = abs(tp_price - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return TPSLLevels(
            tp_price=tp_price,
            sl_price=sl_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            rr_ratio=rr_ratio,
            atr=0.0,
            volatility=0.0,
            method="percentage_based",
            confidence=0.70,
        )
    
    def calculate_microstructure_adjusted(
        self,
        entry_price: float,
        direction: Direction,
        candles: List[Dict],
        orderbook_imbalance: float = 0.0,
        spread_bps: float = 0.0,
    ) -> TPSLLevels:
        """
        Advanced: Adjust TP/SL based on microstructure
        
        Features:
        - Orderbook imbalance consideration
        - Spread-adjusted levels
        - Support/resistance proximity
        """
        # Start with ATR-based
        base_levels = self.calculate_atr_based(entry_price, direction, candles)
        
        # Adjust based on orderbook imbalance
        if abs(orderbook_imbalance) > 0.3:
            # Strong imbalance: tighten SL, extend TP in favorable direction
            if (direction == "long" and orderbook_imbalance > 0) or \
               (direction == "short" and orderbook_imbalance < 0):
                # Favorable imbalance
                tp_adjustment = 1.2  # Extend TP by 20%
                sl_adjustment = 0.9  # Tighten SL by 10%
            else:
                # Unfavorable imbalance
                tp_adjustment = 0.8
                sl_adjustment = 1.1
        else:
            tp_adjustment = 1.0
            sl_adjustment = 1.0
        
        # Adjust based on spread (wider spread = more uncertainty)
        if spread_bps > 10:  # Wide spread
            sl_adjustment *= 1.1  # Wider SL for slippage
        
        # Apply adjustments
        if direction == "long":
            tp_price = entry_price + (base_levels.tp_price - entry_price) * tp_adjustment
            sl_price = entry_price - (entry_price - base_levels.sl_price) * sl_adjustment
        else:
            tp_price = entry_price - (entry_price - base_levels.tp_price) * tp_adjustment
            sl_price = entry_price + (base_levels.sl_price - entry_price) * sl_adjustment
        
        # Recalculate metrics
        risk_amount = abs(entry_price - sl_price)
        reward_amount = abs(tp_price - entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return TPSLLevels(
            tp_price=tp_price,
            sl_price=sl_price,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            rr_ratio=rr_ratio,
            atr=base_levels.atr,
            volatility=base_levels.volatility,
            method="microstructure_adjusted",
            confidence=0.90,
        )
    
    def calculate_smart(
        self,
        entry_price: float,
        direction: Direction,
        candles: Optional[List[Dict]] = None,
        orderbook_imbalance: float = 0.0,
        spread_bps: float = 0.0,
        volatility_regime: str = "normal",
    ) -> TPSLLevels:
        """
        Smart TP/SL calculation - chooses best method based on available data
        
        Priority:
        1. Microstructure-adjusted (if orderbook data available)
        2. ATR-based (if candle data available)
        3. Percentage-based (fallback)
        """
        # Best case: microstructure-adjusted
        if candles and len(candles) >= 20 and abs(orderbook_imbalance) > 0.1:
            levels = self.calculate_microstructure_adjusted(
                entry_price, direction, candles, orderbook_imbalance, spread_bps
            )
        
        # Good case: ATR-based
        elif candles and len(candles) >= 15:
            levels = self.calculate_atr_based(entry_price, direction, candles)
        
        # Fallback: percentage-based
        else:
            # Adjust percentages based on volatility regime
            if volatility_regime == "high":
                tp_pct, sl_pct = 0.03, 0.015  # Wider levels
            elif volatility_regime == "low":
                tp_pct, sl_pct = 0.015, 0.008  # Tighter levels
            else:
                tp_pct, sl_pct = 0.02, 0.01  # Normal
            
            levels = self.calculate_percentage_based(entry_price, direction, tp_pct, sl_pct)
        
        # Validate R:R ratio
        if levels.rr_ratio < self.min_rr_ratio:
            # Adjust TP to meet minimum R:R
            adjustment = self.min_rr_ratio / levels.rr_ratio if levels.rr_ratio > 0 else 1.5
            
            if direction == "long":
                levels.tp_price = entry_price + (levels.tp_price - entry_price) * adjustment
            else:
                levels.tp_price = entry_price - (entry_price - levels.tp_price) * adjustment
            
            levels.reward_amount = abs(levels.tp_price - entry_price)
            levels.rr_ratio = levels.reward_amount / levels.risk_amount if levels.risk_amount > 0 else 0
        
        return levels


__all__ = ["DynamicTPSLCalculator", "ATRCalculator", "TPSLLevels"]
