"""
🛡️ TRADE SAFETY SYSTEM - Complete Trading Safety Controller
Prevents over-trading, manages risk, filters bad signals, protects capital.

This module is the MASTER SAFETY GATE for all trading decisions.
Every signal MUST pass through TradeSafety.should_execute_signal() before execution.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger("trade.safety")


@dataclass
class SafetyConfig:
    """Complete safety configuration"""
    
    # ========== CONFIDENCE THRESHOLDS ==========
    min_confidence: float = 0.70  # 70% minimum (default)
    min_confidence_trending: float = 0.65  # 65% for strong trends
    min_confidence_ranging: float = 0.75  # 75% for ranging markets
    min_confidence_volatile: float = 0.80  # 80% for volatile markets
    
    # ========== RISK FILTERS ==========
    max_spoof_score: float = 0.50  # 50% max spoof risk
    max_risk_penalty: float = 0.80  # 80% max risk penalty
    min_trend_score: float = 0.10  # 10% min trend strength
    max_spread_bps: float = 10.0  # 10 bps max spread
    
    # ========== TIME-BASED LIMITS ==========
    min_trade_interval_seconds: int = 300  # 5 minutes between trades
    min_same_direction_interval_seconds: int = 600  # 10 minutes for same direction
    signal_max_age_seconds: int = 5  # Reject signals older than 5s
    
    # ========== RATE LIMITING ==========
    max_trades_per_hour: int = 4
    max_trades_per_day: int = 20
    max_flips_per_hour: int = 2  # Max position flips
    
    # ========== LOSS PROTECTION ==========
    max_daily_loss_pct: float = 0.05  # 5% max daily loss
    max_consecutive_losses: int = 3  # Stop after 3 losses
    cooldown_after_loss_seconds: int = 900  # 15 min cooldown after loss
    
    # ========== POSITION LIMITS ==========
    max_position_size_btc: float = 0.10  # 0.1 BTC max
    min_position_size_btc: float = 0.001  # 0.001 BTC min
    max_leverage: float = 5.0  # 5x max leverage
    
    # ========== DUPLICATE FILTERING ==========
    duplicate_signal_window_seconds: int = 30  # 30s window
    duplicate_price_threshold_pct: float = 0.001  # 0.1% price difference
    
    # ========== EMERGENCY STOPS ==========
    enable_emergency_stop: bool = True
    emergency_stop_loss_pct: float = 0.10  # 10% total loss = stop all trading


@dataclass
class TradeDecision:
    """Result of safety check"""
    approved: bool
    reason: str
    confidence_required: float
    actual_confidence: float
    warnings: List[str] = field(default_factory=list)
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradeSafety:
    """
    🛡️ Master Safety Controller for All Trading Decisions
    
    Implements 10 safety layers:
    1. Confidence thresholds (regime-adaptive)
    2. Risk filters (spoof, risk_penalty, trend)
    3. Time-based cooldowns
    4. Duplicate signal filtering
    5. Position state validation
    6. Rate limiting (hourly/daily)
    7. Daily loss limits
    8. Consecutive loss protection
    9. Signal age validation
    10. Emergency stop mechanism
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.config = config or SafetyConfig()
        
        # ========== STATE TRACKING ==========
        self.last_trade_time: Optional[datetime] = None
        self.last_signal: Optional[Dict] = None
        self.last_trade_direction: Optional[str] = None
        
        # Position state
        self.current_position: Optional[Dict] = None
        
        # Trade history (sliding windows)
        self.trades_this_hour: deque = deque(maxlen=100)
        self.trades_today: deque = deque(maxlen=500)
        self.flips_this_hour: deque = deque(maxlen=20)
        
        # P&L tracking
        self.daily_pnl: float = 0.0
        self.initial_balance: float = 1000.0  # USDT
        self.day_start: datetime = self._get_day_start()
        
        # Win/Loss streak
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.last_trade_result: Optional[str] = None  # "win" / "loss"
        
        # Cooldown state
        self.cooldown_until: Optional[datetime] = None
        self.cooldown_reason: str = ""
        
        # Emergency stop
        self.emergency_stopped: bool = False
        self.emergency_stop_reason: str = ""
        
        # Statistics
        self.stats = {
            "total_signals_received": 0,
            "total_signals_approved": 0,
            "total_signals_rejected": 0,
            "rejection_reasons": {},
        }
        
        logger.info("🛡️ Trade Safety System initialized")
        logger.info(f"   Min Confidence: {self.config.min_confidence:.0%}")
        logger.info(f"   Min Trade Interval: {self.config.min_trade_interval_seconds}s")
        logger.info(f"   Max Trades/Hour: {self.config.max_trades_per_hour}")
        logger.info(f"   Max Trades/Day: {self.config.max_trades_per_day}")
        logger.info(f"   Max Daily Loss: {self.config.max_daily_loss_pct:.0%}")
    
    # ============================================================
    # MAIN SAFETY GATE
    # ============================================================
    
    def should_execute_signal(self, signal: Dict) -> TradeDecision:
        """
        🛡️ MASTER SAFETY GATE
        
        Determines if a signal should be executed based on ALL safety checks.
        
        Args:
            signal: Dict with keys:
                - direction: "long" / "short" / "flat"
                - confidence: float 0.0-1.0
                - price: float
                - regime: str ("trending_up", "ranging", etc.)
                - spoof_score: float 0.0-1.0
                - risk_penalty: float 0.0-1.0
                - scores: Dict with trend_score, etc.
                - timestamp: datetime or float
        
        Returns:
            TradeDecision with approved/rejected status and detailed reasoning
        """
        self.stats["total_signals_received"] += 1
        now = self._now()
        
        warnings: List[str] = []
        checks_passed: Dict[str, bool] = {}
        metadata: Dict[str, Any] = {}
        
        # Reset daily if needed
        self._reset_daily_if_needed(now)
        
        # Extract signal fields
        direction = signal.get("direction", "flat")
        confidence = float(signal.get("confidence", 0.0))
        price = signal.get("price")
        regime = signal.get("regime", "unknown")
        spoof_score = float(signal.get("spoof_score", 0.0))
        risk_penalty = float(signal.get("risk_penalty", 0.0))
        scores = signal.get("scores", {})
        trend_score = float(scores.get("trend", 0.0))
        signal_timestamp = signal.get("timestamp", now)
        
        # ========== CHECK 1: EMERGENCY STOP ==========
        if self.emergency_stopped:
            checks_passed["emergency_stop"] = False
            return self._reject(
                "emergency_stop_active",
                confidence,
                self.config.min_confidence,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["emergency_stop"] = True
        
        # ========== CHECK 2: COOLDOWN ==========
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds()
            checks_passed["cooldown"] = False
            return self._reject(
                f"cooldown_active: {remaining:.0f}s remaining ({self.cooldown_reason})",
                confidence,
                self.config.min_confidence,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["cooldown"] = True
        
        # ========== CHECK 3: FLAT SIGNAL ==========
        if direction == "flat":
            checks_passed["signal_direction"] = True
            return self._reject(
                "flat_signal_no_action",
                confidence,
                self.config.min_confidence,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["signal_direction"] = True
        
        # ========== CHECK 4: SIGNAL AGE ==========
        if isinstance(signal_timestamp, (int, float)):
            signal_time = datetime.fromtimestamp(signal_timestamp, tz=timezone.utc)
        elif isinstance(signal_timestamp, datetime):
            signal_time = signal_timestamp
        else:
            signal_time = now
        
        signal_age = (now - signal_time).total_seconds()
        if signal_age > self.config.signal_max_age_seconds:
            checks_passed["signal_age"] = False
            return self._reject(
                f"stale_signal: {signal_age:.1f}s old (max {self.config.signal_max_age_seconds}s)",
                confidence,
                self.config.min_confidence,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["signal_age"] = True
        metadata["signal_age_seconds"] = signal_age
        
        # ========== CHECK 5: REGIME-ADAPTIVE CONFIDENCE ==========
        min_conf_required = self._get_min_confidence_for_regime(regime)
        
        if confidence < min_conf_required:
            checks_passed["confidence"] = False
            return self._reject(
                f"confidence_too_low: {confidence:.1%} < {min_conf_required:.1%} (regime={regime})",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["confidence"] = True
        metadata["min_confidence_required"] = min_conf_required
        
        # ========== CHECK 6: SPOOF DETECTION ==========
        if spoof_score > self.config.max_spoof_score:
            checks_passed["spoof_filter"] = False
            return self._reject(
                f"high_spoof_risk: {spoof_score:.0%} > {self.config.max_spoof_score:.0%}",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["spoof_filter"] = True
        if spoof_score > 0.3:
            warnings.append(f"Moderate spoof risk: {spoof_score:.0%}")
        
        # ========== CHECK 7: RISK PENALTY ==========
        if risk_penalty > self.config.max_risk_penalty:
            checks_passed["risk_penalty_filter"] = False
            return self._reject(
                f"high_risk_penalty: {risk_penalty:.0%} > {self.config.max_risk_penalty:.0%}",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["risk_penalty_filter"] = True
        if risk_penalty > 0.6:
            warnings.append(f"Elevated risk penalty: {risk_penalty:.0%}")
        
        # ========== CHECK 8: TREND STRENGTH ==========
        if abs(trend_score) < self.config.min_trend_score:
            checks_passed["trend_strength"] = False
            warnings.append(f"Weak trend: {abs(trend_score):.1%}")
            # Don't reject, just warn
        checks_passed["trend_strength"] = True
        
        # ========== CHECK 9: PRICE VALIDITY ==========
        if price is None or price <= 0:
            checks_passed["price_validity"] = False
            return self._reject(
                "invalid_price",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["price_validity"] = True
        
        # ========== CHECK 10: DUPLICATE SIGNAL FILTERING ==========
        if self._is_duplicate_signal(signal, now):
            checks_passed["duplicate_filter"] = False
            return self._reject(
                f"duplicate_signal: same {direction} within {self.config.duplicate_signal_window_seconds}s",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["duplicate_filter"] = True
        
        # ========== CHECK 11: TIME-BASED COOLDOWN ==========
        if self.last_trade_time:
            time_since_last = (now - self.last_trade_time).total_seconds()
            min_interval = self.config.min_trade_interval_seconds
            
            # Extra interval for same direction
            if direction == self.last_trade_direction:
                min_interval = self.config.min_same_direction_interval_seconds
            
            if time_since_last < min_interval:
                checks_passed["time_cooldown"] = False
                remaining = min_interval - time_since_last
                return self._reject(
                    f"time_cooldown: {remaining:.0f}s remaining (last trade {time_since_last:.0f}s ago)",
                    confidence,
                    min_conf_required,
                    warnings,
                    checks_passed,
                    metadata
                )
        checks_passed["time_cooldown"] = True
        
        # ========== CHECK 12: POSITION STATE ==========
        if self.current_position is not None:
            current_dir = self.current_position.get("direction")
            if current_dir == direction:
                checks_passed["position_state"] = False
                return self._reject(
                    f"position_already_open: {direction} position exists",
                    confidence,
                    min_conf_required,
                    warnings,
                    checks_passed,
                    metadata
                )
            else:
                # Flipping position - check flip limits
                flip_count = self._count_recent_flips(now, hours=1)
                if flip_count >= self.config.max_flips_per_hour:
                    checks_passed["flip_limit"] = False
                    return self._reject(
                        f"flip_limit_reached: {flip_count} flips this hour (max {self.config.max_flips_per_hour})",
                        confidence,
                        min_conf_required,
                        warnings,
                        checks_passed,
                        metadata
                    )
                warnings.append(f"Will flip position: {current_dir} → {direction}")
        checks_passed["position_state"] = True
        
        # ========== CHECK 13: HOURLY TRADE LIMIT ==========
        hourly_count = self._count_recent_trades(now, hours=1)
        if hourly_count >= self.config.max_trades_per_hour:
            checks_passed["hourly_limit"] = False
            return self._reject(
                f"hourly_limit_reached: {hourly_count}/{self.config.max_trades_per_hour} trades this hour",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["hourly_limit"] = True
        metadata["trades_this_hour"] = hourly_count
        
        # ========== CHECK 14: DAILY TRADE LIMIT ==========
        daily_count = len(self.trades_today)
        if daily_count >= self.config.max_trades_per_day:
            checks_passed["daily_limit"] = False
            return self._reject(
                f"daily_limit_reached: {daily_count}/{self.config.max_trades_per_day} trades today",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["daily_limit"] = True
        metadata["trades_today"] = daily_count
        
        # ========== CHECK 15: DAILY LOSS LIMIT ==========
        max_daily_loss = self.initial_balance * self.config.max_daily_loss_pct
        if self.daily_pnl < -max_daily_loss:
            checks_passed["daily_loss_limit"] = False
            self._trigger_emergency_stop(f"Daily loss limit reached: ${abs(self.daily_pnl):.2f}")
            return self._reject(
                f"daily_loss_limit_exceeded: ${abs(self.daily_pnl):.2f} > ${max_daily_loss:.2f}",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["daily_loss_limit"] = True
        metadata["daily_pnl"] = self.daily_pnl
        
        # ========== CHECK 16: CONSECUTIVE LOSSES ==========
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            checks_passed["consecutive_losses"] = False
            return self._reject(
                f"consecutive_loss_limit: {self.consecutive_losses} losses in a row (max {self.config.max_consecutive_losses})",
                confidence,
                min_conf_required,
                warnings,
                checks_passed,
                metadata
            )
        checks_passed["consecutive_losses"] = True
        metadata["consecutive_losses"] = self.consecutive_losses
        
        # ========== ALL CHECKS PASSED ✅ ==========
        self.stats["total_signals_approved"] += 1
        
        return TradeDecision(
            approved=True,
            reason="all_safety_checks_passed",
            confidence_required=min_conf_required,
            actual_confidence=confidence,
            warnings=warnings,
            checks_passed=checks_passed,
            metadata=metadata
        )
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _reject(
        self,
        reason: str,
        confidence: float,
        min_conf: float,
        warnings: List[str],
        checks_passed: Dict[str, bool],
        metadata: Dict[str, Any]
    ) -> TradeDecision:
        """Create rejection decision"""
        self.stats["total_signals_rejected"] += 1
        
        # Track rejection reasons
        if reason not in self.stats["rejection_reasons"]:
            self.stats["rejection_reasons"][reason] = 0
        self.stats["rejection_reasons"][reason] += 1
        
        return TradeDecision(
            approved=False,
            reason=reason,
            confidence_required=min_conf,
            actual_confidence=confidence,
            warnings=warnings,
            checks_passed=checks_passed,
            metadata=metadata
        )
    
    def _get_min_confidence_for_regime(self, regime: str) -> float:
        """Get minimum confidence based on market regime"""
        regime_lower = regime.lower()
        
        if "trend" in regime_lower or "breakout" in regime_lower:
            return self.config.min_confidence_trending
        elif "range" in regime_lower or "consolidation" in regime_lower:
            return self.config.min_confidence_ranging
        elif "volatile" in regime_lower or "chaos" in regime_lower:
            return self.config.min_confidence_volatile
        else:
            return self.config.min_confidence
    
    def _is_duplicate_signal(self, signal: Dict, now: datetime) -> bool:
        """Check if signal is duplicate of recent signal"""
        if not self.last_signal:
            return False
        
        # Check time window
        if not self.last_trade_time:
            return False
        
        time_diff = (now - self.last_trade_time).total_seconds()
        if time_diff > self.config.duplicate_signal_window_seconds:
            return False
        
        # Check direction
        if signal.get("direction") != self.last_signal.get("direction"):
            return False
        
        # Check price similarity
        current_price = signal.get("price")
        last_price = self.last_signal.get("price")
        
        if current_price and last_price:
            price_diff_pct = abs(current_price - last_price) / last_price
            if price_diff_pct < self.config.duplicate_price_threshold_pct:
                return True
        
        return False
    
    def _count_recent_trades(self, now: datetime, hours: int) -> int:
        """Count trades in last N hours"""
        cutoff = now - timedelta(hours=hours)
        return sum(1 for t in self.trades_this_hour if t["time"] > cutoff)
    
    def _count_recent_flips(self, now: datetime, hours: int) -> int:
        """Count position flips in last N hours"""
        cutoff = now - timedelta(hours=hours)
        return sum(1 for f in self.flips_this_hour if f["time"] > cutoff)
    
    def _now(self) -> datetime:
        """Get current time in UTC"""
        return datetime.now(timezone.utc)
    
    def _get_day_start(self) -> datetime:
        """Get start of current day"""
        now = self._now()
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _reset_daily_if_needed(self, now: datetime) -> None:
        """Reset daily counters if new day"""
        if now.date() > self.day_start.date():
            logger.info(f"🔄 Daily reset: PnL was ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.trades_today.clear()
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.day_start = self._get_day_start()
            
            # Clear emergency stop on new day
            if self.emergency_stopped:
                logger.info("🔓 Emergency stop cleared (new day)")
                self.emergency_stopped = False
                self.emergency_stop_reason = ""
    
    def _trigger_emergency_stop(self, reason: str) -> None:
        """Activate emergency stop"""
        self.emergency_stopped = True
        self.emergency_stop_reason = reason
        logger.critical(f"🚨 EMERGENCY STOP ACTIVATED: {reason}")
    
    # ============================================================
    # TRADE LIFECYCLE MANAGEMENT
    # ============================================================
    
    def record_trade(self, trade: Dict) -> None:
        """
        Record a trade for tracking.
        
        Args:
            trade: Dict with keys:
                - direction: str
                - price: float
                - size: float
                - timestamp: datetime
        """
        now = self._now()
        trade["time"] = now
        
        self.trades_this_hour.append(trade)
        self.trades_today.append(trade)
        self.last_trade_time = now
        self.last_signal = trade
        self.last_trade_direction = trade.get("direction")
        
        # Track position
        self.current_position = {
            "direction": trade.get("direction"),
            "size": trade.get("size"),
            "entry_price": trade.get("price"),
            "opened_at": now,
        }
        
        # Track flips
        if trade.get("is_flip", False):
            self.flips_this_hour.append({"time": now})
        
        logger.info(
            f"📝 Trade recorded: {trade.get('direction')} @ {trade.get('price'):.2f} "
            f"| Trades today: {len(self.trades_today)}, hour: {self._count_recent_trades(now, 1)}"
        )
    
    def record_trade_result(self, pnl: float, is_win: bool) -> None:
        """
        Record trade result (P&L).
        
        Args:
            pnl: Profit/Loss in USDT
            is_win: True if profitable
        """
        self.daily_pnl += pnl
        
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.last_trade_result = "win"
            logger.info(f"✅ Win recorded: ${pnl:.2f} | Streak: {self.consecutive_wins} wins | Daily PnL: ${self.daily_pnl:.2f}")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_trade_result = "loss"
            logger.warning(f"❌ Loss recorded: ${pnl:.2f} | Streak: {self.consecutive_losses} losses | Daily PnL: ${self.daily_pnl:.2f}")
            
            # Enter cooldown after loss
            if self.config.cooldown_after_loss_seconds > 0:
                self._enter_cooldown(
                    self.config.cooldown_after_loss_seconds,
                    f"Loss cooldown ({self.consecutive_losses} losses)"
                )
    
    def close_position(self, close_price: float, reason: str = "") -> None:
        """Close current position"""
        if self.current_position:
            logger.info(f"❌ Position closed: {self.current_position.get('direction')} @ {close_price:.2f} | Reason: {reason}")
            self.current_position = None
    
    def _enter_cooldown(self, seconds: int, reason: str) -> None:
        """Enter cooldown period"""
        self.cooldown_until = self._now() + timedelta(seconds=seconds)
        self.cooldown_reason = reason
        logger.warning(f"⏸️ Cooldown activated: {seconds}s | Reason: {reason}")
    
    def exit_cooldown(self) -> None:
        """Manually exit cooldown"""
        if self.cooldown_until:
            logger.info("▶️ Cooldown manually cleared")
            self.cooldown_until = None
            self.cooldown_reason = ""
    
    def reset_emergency_stop(self) -> None:
        """Manually reset emergency stop"""
        if self.emergency_stopped:
            logger.warning("🔓 Emergency stop manually cleared")
            self.emergency_stopped = False
            self.emergency_stop_reason = ""
    
    # ============================================================
    # STATUS & REPORTING
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current safety system status"""
        now = self._now()
        
        cooldown_remaining = 0
        if self.cooldown_until and now < self.cooldown_until:
            cooldown_remaining = (self.cooldown_until - now).total_seconds()
        
        approval_rate = 0.0
        if self.stats["total_signals_received"] > 0:
            approval_rate = self.stats["total_signals_approved"] / self.stats["total_signals_received"]
        
        return {
            "active": True,
            "emergency_stopped": self.emergency_stopped,
            "emergency_stop_reason": self.emergency_stop_reason,
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining_seconds": cooldown_remaining,
            "cooldown_reason": self.cooldown_reason,
            "trades_this_hour": self._count_recent_trades(now, 1),
            "trades_today": len(self.trades_today),
            "flips_this_hour": self._count_recent_flips(now, 1),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / self.initial_balance) if self.initial_balance > 0 else 0.0,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_result": self.last_trade_result,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "current_position": self.current_position,
            "approval_rate": approval_rate,
            "total_signals_received": self.stats["total_signals_received"],
            "total_signals_approved": self.stats["total_signals_approved"],
            "total_signals_rejected": self.stats["total_signals_rejected"],
            "top_rejection_reasons": sorted(
                self.stats["rejection_reasons"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }
    
    def log_status(self) -> None:
        """Log current status"""
        status = self.get_status()
        logger.info(
            f"🛡️ Safety Status: "
            f"Trades: {status['trades_this_hour']}/hour, {status['trades_today']}/day | "
            f"PnL: ${status['daily_pnl']:.2f} ({status['daily_pnl_pct']:.1%}) | "
            f"Streak: W={status['consecutive_wins']} L={status['consecutive_losses']} | "
            f"Approval: {status['approval_rate']:.0%} "
            f"({status['total_signals_approved']}/{status['total_signals_received']})"
        )

