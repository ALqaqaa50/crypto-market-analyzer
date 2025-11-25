# okx_stream_hunter/integrations/risk_manager.py
"""
🔥 Risk Manager - Professional Risk Management System (UPGRADED)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional, Tuple

# Import advanced TP/SL calculator
try:
    from ..modules.tpsl.calculator import DynamicTPSLCalculator, TPSLLevels
except ImportError:
    DynamicTPSLCalculator = None
    TPSLLevels = None

logger = logging.getLogger("risk.manager")

Direction = Literal["long", "short", "flat"]


class RiskManager:
    """Enhanced Risk Manager for Phase 2"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_losses = 0
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.10)
        self.max_daily_trades = config.get('max_daily_trades', 20)
        self.risk_locked = False
        self.last_reset = datetime.now().date()
        
        logger.info("🛡️ Risk Manager initialized")
        logger.info(f"   Max Daily Drawdown: {self.max_daily_drawdown:.1%}")
        logger.info(f"   Max Daily Trades: {self.max_daily_trades}")
    
    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        self._check_daily_reset()
        
        if self.risk_locked:
            logger.warning("🚫 Trading LOCKED due to risk limits")
            return False
        
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"🚫 Max daily trades reached: {self.daily_trades}")
            return False
        
        initial_balance = self.config.get('initial_balance', 10000)
        if self.daily_pnl < -(initial_balance * self.max_daily_drawdown):
            logger.warning(f"🚫 Max daily drawdown reached: ${self.daily_pnl:.2f}")
            self.risk_locked = True
            return False
        
        return True
    
    def register_trade(self, trade: Dict):
        """Register a new trade"""
        self.daily_trades += 1
        
        pnl = trade.get('pnl', 0.0)
        if pnl != 0:
            self.daily_pnl += pnl
            if pnl < 0:
                self.daily_losses += 1
        
        logger.info(f"📊 Trade registered: Daily PnL=${self.daily_pnl:.2f}, Trades={self.daily_trades}")
    
    def _check_daily_reset(self):
        """Reset daily counters at midnight"""
        today = datetime.now().date()
        if today > self.last_reset:
            logger.info("🔄 Daily risk counters reset")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_losses = 0
            self.risk_locked = False
            self.last_reset = today
    
    def get_stats(self) -> Dict:
        """Get risk statistics"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_losses': self.daily_losses,
            'risk_locked': self.risk_locked,
            'trades_remaining': max(0, self.max_daily_trades - self.daily_trades),
            'drawdown_remaining': self.max_daily_drawdown - abs(self.daily_pnl) / self.config.get('initial_balance', 10000)
        }


logger = logging.getLogger("risk.manager")

Direction = Literal["long", "short", "flat"]


@dataclass
class RiskConfig:
    """Risk management configuration"""
    
    # Account management
    account_balance: float = 1000.0  # Total account balance in USDT
    max_risk_per_trade_pct: float = 0.01  # 1% max risk per trade
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_position_size_pct: float = 0.10  # 10% max position size
    
    # Position limits
    max_open_positions: int = 3
    max_leverage: float = 10.0
    min_position_size: float = 0.001  # Minimum BTC size
    
    # Risk/Reward
    min_rr_ratio: float = 1.5  # Minimum 1.5:1 R:R
    default_rr_ratio: float = 2.0  # Default 2:1 R:R
    
    # Volatility-based adjustments
    enable_volatility_adjustment: bool = True
    low_vol_multiplier: float = 1.5  # Increase size in low vol
    high_vol_multiplier: float = 0.5  # Decrease size in high vol
    high_vol_threshold: float = 0.03  # 3% = high volatility
    low_vol_threshold: float = 0.01  # 1% = low volatility
    
    # Correlation & hedging
    enable_correlation_check: bool = True
    max_correlated_exposure: float = 0.15  # 15% max for correlated pairs
    
    # Drawdown protection
    enable_drawdown_protection: bool = True
    reduce_size_after_loss: bool = True
    loss_reduction_factor: float = 0.5  # Halve size after loss
    win_increase_factor: float = 1.2  # Increase 20% after win
    consecutive_losses_limit: int = 3  # Stop after 3 losses


@dataclass
class TradeRisk:
    """Risk assessment for a single trade"""
    
    approved: bool
    position_size: float
    risk_amount: float
    risk_pct: float
    
    entry_price: float
    tp_price: Optional[float]
    sl_price: Optional[float]
    
    rr_ratio: float
    expected_profit: float
    expected_loss: float
    
    reason: str
    warnings: List[str]
    adjustments: Dict[str, float]


class RiskManager:
    """
    🔥 Professional Risk Management System
    
    Features:
    - Dynamic position sizing based on risk %
    - Kelly Criterion for optimal sizing
    - Volatility-adjusted sizing
    - Drawdown protection
    - Daily loss limits
    - Win/Loss streak tracking
    - R:R ratio validation
    - Multi-position exposure limits
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        
        # Daily tracking
        self.daily_pnl: float = 0.0
        self.daily_reset_time: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        # Streak tracking
        self.consecutive_wins: int = 0
        self.consecutive_losses: int = 0
        self.last_trade_result: Optional[str] = None  # "win" / "loss"
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0.0
        
        # Current exposure
        self.open_positions: Dict[str, Dict] = {}
        
        # Advanced TP/SL calculator
        self.tpsl_calculator = DynamicTPSLCalculator() if DynamicTPSLCalculator else None
    
    # ============================================================
    # Core Risk Assessment
    # ============================================================
    
    def assess_trade(
        self,
        symbol: str,
        direction: Direction,
        entry_price: float,
        sl_price: float,
        tp_price: Optional[float] = None,
        confidence: float = 0.5,
        volatility: Optional[float] = None,
    ) -> TradeRisk:
        """
        🔥 Assess trade risk and calculate optimal position size.
        
        Returns TradeRisk with approved/rejected status and sizing.
        """
        warnings: List[str] = []
        adjustments: Dict[str, float] = {}
        
        # 1. Check daily loss limit
        self._reset_daily_if_needed()
        
        if self.daily_pnl <= -self.config.max_daily_loss_pct * self.config.account_balance:
            return TradeRisk(
                approved=False,
                position_size=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                rr_ratio=0.0,
                expected_profit=0.0,
                expected_loss=0.0,
                reason="daily_loss_limit_reached",
                warnings=["Daily loss limit reached"],
                adjustments={},
            )
        
        # 2. Check consecutive losses
        if (
            self.config.enable_drawdown_protection
            and self.consecutive_losses >= self.config.consecutive_losses_limit
        ):
            return TradeRisk(
                approved=False,
                position_size=0.0,
                risk_amount=0.0,
                risk_pct=0.0,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                rr_ratio=0.0,
                expected_profit=0.0,
                expected_loss=0.0,
                reason="consecutive_losses_limit",
                warnings=[f"{self.consecutive_losses} consecutive losses - cooling off"],
                adjustments={},
            )
        
        # 3. Calculate SL distance and R:R
        sl_distance_pct = abs(entry_price - sl_price) / entry_price
        
        if tp_price is None:
            # Calculate TP based on default R:R
            tp_distance_pct = sl_distance_pct * self.config.default_rr_ratio
            if direction == "long":
                tp_price = entry_price * (1.0 + tp_distance_pct)
            else:
                tp_price = entry_price * (1.0 - tp_distance_pct)
        
        tp_distance_pct = abs(tp_price - entry_price) / entry_price
        rr_ratio = tp_distance_pct / sl_distance_pct if sl_distance_pct > 0 else 0.0
        
        # 4. Validate R:R ratio
        if rr_ratio < self.config.min_rr_ratio:
            warnings.append(f"Low R:R ratio: {rr_ratio:.2f} < {self.config.min_rr_ratio}")
            # Could reject, but let's just warn
        
        # 5. Calculate base position size (risk-based)
        max_risk_amount = self.config.account_balance * self.config.max_risk_per_trade_pct
        position_size = max_risk_amount / (sl_distance_pct * entry_price)
        adjustments["base_size"] = position_size
        
        # 6. Apply confidence adjustment
        position_size *= confidence
        adjustments["confidence_adjusted"] = position_size
        
        # 7. Apply volatility adjustment
        if self.config.enable_volatility_adjustment and volatility is not None:
            if volatility > self.config.high_vol_threshold:
                position_size *= self.config.high_vol_multiplier
                warnings.append(f"High volatility ({volatility:.2%}) - reduced size")
                adjustments["volatility_adjusted"] = position_size
            elif volatility < self.config.low_vol_threshold:
                position_size *= self.config.low_vol_multiplier
                adjustments["volatility_adjusted"] = position_size
        
        # 8. Apply win/loss streak adjustment
        if self.config.reduce_size_after_loss and self.last_trade_result == "loss":
            position_size *= self.config.loss_reduction_factor
            warnings.append("Size reduced after loss")
            adjustments["loss_adjusted"] = position_size
        
        if self.last_trade_result == "win" and self.consecutive_wins >= 2:
            position_size *= self.config.win_increase_factor
            adjustments["win_adjusted"] = position_size
        
        # 9. Apply position size limits
        max_position_value = self.config.account_balance * self.config.max_position_size_pct
        max_size_by_value = max_position_value / entry_price
        
        if position_size > max_size_by_value:
            position_size = max_size_by_value
            warnings.append("Size capped at max position %")
            adjustments["max_position_capped"] = position_size
        
        if position_size < self.config.min_position_size:
            position_size = self.config.min_position_size
            warnings.append("Size increased to minimum")
            adjustments["min_position_adjusted"] = position_size
        
        # 10. Check leverage
        position_value = position_size * entry_price
        leverage = position_value / self.config.account_balance
        
        if leverage > self.config.max_leverage:
            position_size = (self.config.max_leverage * self.config.account_balance) / entry_price
            warnings.append(f"Leverage capped at {self.config.max_leverage}x")
            adjustments["leverage_capped"] = position_size
        
        # 11. Calculate expected P&L
        expected_loss = position_size * sl_distance_pct * entry_price
        expected_profit = position_size * tp_distance_pct * entry_price
        risk_pct = expected_loss / self.config.account_balance
        
        # 12. Final approval
        approved = True
        reason = "approved"
        
        if risk_pct > self.config.max_risk_per_trade_pct * 1.5:  # 1.5x buffer
            approved = False
            reason = "risk_too_high"
            warnings.append(f"Risk {risk_pct:.2%} exceeds max")
        
        if position_size <= 0:
            approved = False
            reason = "invalid_size"
        
        return TradeRisk(
            approved=approved,
            position_size=round(position_size, 4),
            risk_amount=expected_loss,
            risk_pct=risk_pct,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            rr_ratio=rr_ratio,
            expected_profit=expected_profit,
            expected_loss=expected_loss,
            reason=reason,
            warnings=warnings,
            adjustments=adjustments,
        )
    
    # ============================================================
    # Kelly Criterion (Optional Advanced Sizing)
    # ============================================================
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Formula: f = (bp - q) / b
        where:
        - f = fraction of capital to bet
        - b = odds (avg_win / avg_loss)
        - p = win probability
        - q = loss probability (1 - p)
        
        Returns fraction of capital (0.0 - 1.0).
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Apply fractional Kelly (safer)
        fractional_kelly = kelly_fraction * 0.25  # Use 1/4 Kelly for safety
        
        return max(0.0, min(fractional_kelly, 0.2))  # Cap at 20%
    
    def suggest_size_with_kelly(
        self,
        entry_price: float,
        sl_distance_pct: float,
    ) -> float:
        """Suggest position size using Kelly criterion based on historical performance"""
        
        if self.total_trades < 10:
            # Not enough data, use conservative sizing
            return self.config.min_position_size
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0.5
        
        # Estimate avg win/loss (simplified)
        avg_win = abs(self.total_pnl / max(1, self.winning_trades)) if self.winning_trades > 0 else 0.0
        avg_loss = abs(self.total_pnl / max(1, self.losing_trades)) if self.losing_trades > 0 else 0.0
        
        if avg_loss <= 0:
            avg_loss = sl_distance_pct * entry_price
        
        kelly_fraction = self.kelly_criterion(win_rate, avg_win, avg_loss)
        
        position_value = self.config.account_balance * kelly_fraction
        position_size = position_value / entry_price
        
        logger.info(
            f"Kelly sizing: win_rate={win_rate:.2%}, kelly={kelly_fraction:.2%}, "
            f"size={position_size:.4f}"
        )
        
        return position_size
    
    # ============================================================
    # Advanced TP/SL Calculation (NEW)
    # ============================================================
    
    def calculate_smart_tpsl(
        self,
        entry_price: float,
        direction: Direction,
        candles: Optional[List[Dict]] = None,
        orderbook_imbalance: float = 0.0,
        spread_bps: float = 0.0,
        volatility_regime: str = "normal",
    ) -> Tuple[float, float]:
        """
        🔥 Calculate optimal TP/SL using advanced algorithms
        
        Returns: (tp_price, sl_price)
        """
        if self.tpsl_calculator and candles:
            levels = self.tpsl_calculator.calculate_smart(
                entry_price=entry_price,
                direction=direction,
                candles=candles,
                orderbook_imbalance=orderbook_imbalance,
                spread_bps=spread_bps,
                volatility_regime=volatility_regime,
            )
            
            logger.info(
                f"Smart TP/SL: method={levels.method}, R:R={levels.rr_ratio:.2f}, "
                f"ATR={levels.atr:.2f}, volatility={levels.volatility:.4f}"
            )
            
            return levels.tp_price, levels.sl_price
        
        # Fallback to simple calculation
        return self._calculate_tp_sl(entry_price, direction, self.config.default_rr_ratio)
    
    # ============================================================
    # Position Tracking
    # ============================================================
    
    def register_position(
        self,
        symbol: str,
        direction: Direction,
        size: float,
        entry_price: float,
    ) -> None:
        """Register an open position for exposure tracking"""
        self.open_positions[symbol] = {
            "direction": direction,
            "size": size,
            "entry_price": entry_price,
            "value": size * entry_price,
        }
        logger.info(f"Position registered: {symbol} {direction} {size}")
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        is_win: bool,
    ) -> None:
        """
        Close position and update stats.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            is_win: True if trade was profitable
        """
        if symbol not in self.open_positions:
            logger.warning(f"Position {symbol} not found in risk manager")
            return
        
        pos = self.open_positions[symbol]
        
        # Calculate P&L
        if pos["direction"] == "long":
            pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            pnl_pct = (pos["entry_price"] - exit_price) / pos["entry_price"]
        
        pnl = pnl_pct * pos["value"]
        
        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl
        self.daily_pnl += pnl
        
        if is_win or pnl > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.last_trade_result = "win"
            logger.info(f"✅ Win: {symbol} PnL={pnl:.2f} (streak: {self.consecutive_wins})")
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_trade_result = "loss"
            logger.warning(f"❌ Loss: {symbol} PnL={pnl:.2f} (streak: {self.consecutive_losses})")
        
        # Remove from open positions
        del self.open_positions[symbol]
    
    def get_total_exposure(self) -> float:
        """Get total exposure across all open positions"""
        return sum(pos["value"] for pos in self.open_positions.values())
    
    def get_exposure_pct(self) -> float:
        """Get exposure as % of account balance"""
        return self.get_total_exposure() / self.config.account_balance
    
    # ============================================================
    # Daily Reset
    # ============================================================
    
    def _reset_daily_if_needed(self) -> None:
        """Reset daily stats at midnight UTC"""
        now = datetime.now(timezone.utc)
        if now.date() > self.daily_reset_time.date():
            logger.info(f"Daily reset: PnL was {self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # ============================================================
    # Reporting
    # ============================================================
    
    def get_stats(self) -> Dict:
        """Get current risk manager statistics"""
        win_rate = (
            self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0
        )
        
        return {
            "account_balance": self.config.account_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.config.account_balance,
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "open_positions": len(self.open_positions),
            "total_exposure": self.get_total_exposure(),
            "exposure_pct": self.get_exposure_pct(),
            "last_trade": self.last_trade_result,
        }
    
    def log_stats(self) -> None:
        """Log current stats"""
        stats = self.get_stats()
        logger.info(
            f"Risk Stats: Balance={stats['account_balance']:.2f}, "
            f"Daily PnL={stats['daily_pnl']:.2f} ({stats['daily_pnl_pct']:.2%}), "
            f"Total PnL={stats['total_pnl']:.2f}, "
            f"Win Rate={stats['win_rate']:.1%} ({stats['winning_trades']}/{stats['total_trades']}), "
            f"Streak: W={stats['consecutive_wins']} L={stats['consecutive_losses']}, "
            f"Exposure={stats['exposure_pct']:.1%}"
        )
