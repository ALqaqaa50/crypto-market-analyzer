# okx_stream_hunter/core/trading_engine.py
"""
🔥 Trading Engine - Advanced State Machine & Market Regime Handler
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Literal, Optional, Any, Callable

from ..integrations.position_manager import PositionManager, PositionManagerConfig
from ..integrations.risk_manager import RiskManager, RiskConfig
from ..integrations.trade_executor import TradeExecutor
from .ai_brain import AIBrain, Signal

logger = logging.getLogger("trading.engine")

Direction = Literal["long", "short", "flat"]


# ============================================================
# Trading States
# ============================================================

class TradingState(Enum):
    """Trading engine states"""
    IDLE = "idle"                    # No position, waiting for signal
    ANALYZING = "analyzing"          # AI analyzing market
    PENDING_ENTRY = "pending_entry"  # Signal received, preparing entry
    IN_POSITION = "in_position"      # Position open, monitoring
    PENDING_EXIT = "pending_exit"    # Exit signal, closing position
    COOLDOWN = "cooldown"            # After loss, cooling down
    PAUSED = "paused"                # Manually paused
    ERROR = "error"                  # Error state


class MarketRegime(Enum):
    """Market regimes for adaptive strategy"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


# ============================================================
# Trading Engine Configuration
# ============================================================

@dataclass
class TradingEngineConfig:
    """Trading engine configuration"""
    
    # State machine
    enable_state_machine: bool = True
    cooldown_after_loss_seconds: int = 300  # 5 min cooldown after loss
    max_analysis_age_seconds: int = 10      # Reject stale signals
    
    # Market regime adaptation
    adapt_to_regime: bool = True
    
    # Trending market settings
    trending_min_confidence: float = 0.35
    trending_position_size_multiplier: float = 1.2
    trending_tp_multiplier: float = 1.5  # Wider TPs in trends
    
    # Ranging market settings
    ranging_min_confidence: float = 0.50  # Higher bar for ranging
    ranging_position_size_multiplier: float = 0.8
    ranging_tp_multiplier: float = 0.8  # Tighter TPs in range
    
    # Volatile market settings
    volatile_min_confidence: float = 0.60  # Much higher bar
    volatile_position_size_multiplier: float = 0.5  # Smaller size
    volatile_sl_multiplier: float = 1.5  # Wider SL for noise
    
    # Safety limits
    max_trades_per_hour: int = 10
    max_trades_per_day: int = 50
    require_risk_approval: bool = True


# ============================================================
# Trading Engine
# ============================================================

class TradingEngine:
    """
    🔥 Advanced Trading Engine with State Machine
    
    Features:
    - State machine for trade lifecycle
    - Market regime detection & adaptation
    - Integration with AI Brain, Risk Manager, Position Manager
    - Automatic position management
    - Performance tracking
    - Event callbacks for extensibility
    """
    
    def __init__(
        self,
        ai_brain: AIBrain,
        risk_manager: Optional[RiskManager] = None,
        position_manager: Optional[PositionManager] = None,
        trade_executor: Optional[TradeExecutor] = None,
        config: Optional[TradingEngineConfig] = None,
        on_state_change: Optional[Callable] = None,
        on_trade_opened: Optional[Callable] = None,
        on_trade_closed: Optional[Callable] = None,
    ):
        self.ai_brain = ai_brain
        self.risk_manager = risk_manager or RiskManager()
        self.position_manager = position_manager or PositionManager()
        self.trade_executor = trade_executor or TradeExecutor(
            position_manager=self.position_manager
        )
        self.config = config or TradingEngineConfig()
        
        # State
        self.state: TradingState = TradingState.IDLE
        self.market_regime: MarketRegime = MarketRegime.UNKNOWN
        
        # Cooldown tracking
        self.cooldown_until: Optional[datetime] = None
        
        # Rate limiting
        self.trades_this_hour: int = 0
        self.trades_this_day: int = 0
        self.hour_reset_time: datetime = datetime.now(timezone.utc)
        self.day_reset_time: datetime = datetime.now(timezone.utc)
        
        # Last signal
        self.last_signal: Optional[Signal] = None
        self.last_decision_time: Optional[datetime] = None
        
        # Callbacks
        self.on_state_change = on_state_change
        self.on_trade_opened = on_trade_opened
        self.on_trade_closed = on_trade_closed
        
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
    
    # ============================================================
    # State Machine
    # ============================================================
    
    def _change_state(self, new_state: TradingState, reason: str = "") -> None:
        """Change trading state with logging and callback"""
        old_state = self.state
        self.state = new_state
        
        logger.info(f"State: {old_state.value} → {new_state.value} | {reason}")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state, reason)
            except Exception as e:
                logger.error(f"Error in on_state_change callback: {e}")
    
    def _detect_market_regime(self, signal: Signal) -> MarketRegime:
        """Detect market regime from AI signal"""
        regime_str = signal.get("regime", "unknown").lower()
        
        if "trend_up" in regime_str or "trending_up" in regime_str:
            return MarketRegime.TRENDING_UP
        elif "trend_down" in regime_str or "trending_down" in regime_str:
            return MarketRegime.TRENDING_DOWN
        elif "range" in regime_str or "ranging" in regime_str:
            return MarketRegime.RANGING
        elif "volatile" in regime_str or "chaos" in regime_str:
            return MarketRegime.VOLATILE
        else:
            return MarketRegime.UNKNOWN
    
    def _get_regime_params(self) -> Dict[str, float]:
        """Get trading parameters for current market regime"""
        cfg = self.config
        
        if self.market_regime == MarketRegime.TRENDING_UP or self.market_regime == MarketRegime.TRENDING_DOWN:
            return {
                "min_confidence": cfg.trending_min_confidence,
                "size_multiplier": cfg.trending_position_size_multiplier,
                "tp_multiplier": cfg.trending_tp_multiplier,
                "sl_multiplier": 1.0,
            }
        elif self.market_regime == MarketRegime.RANGING:
            return {
                "min_confidence": cfg.ranging_min_confidence,
                "size_multiplier": cfg.ranging_position_size_multiplier,
                "tp_multiplier": cfg.ranging_tp_multiplier,
                "sl_multiplier": 1.0,
            }
        elif self.market_regime == MarketRegime.VOLATILE:
            return {
                "min_confidence": cfg.volatile_min_confidence,
                "size_multiplier": cfg.volatile_position_size_multiplier,
                "tp_multiplier": 1.0,
                "sl_multiplier": cfg.volatile_sl_multiplier,
            }
        else:
            return {
                "min_confidence": 0.40,
                "size_multiplier": 1.0,
                "tp_multiplier": 1.0,
                "sl_multiplier": 1.0,
            }
    
    # ============================================================
    # Rate Limiting
    # ============================================================
    
    def _check_rate_limits(self) -> tuple[bool, str]:
        """Check if rate limits are exceeded"""
        now = datetime.now(timezone.utc)
        
        # Reset counters if needed
        if (now - self.hour_reset_time).total_seconds() >= 3600:
            self.trades_this_hour = 0
            self.hour_reset_time = now
        
        if (now - self.day_reset_time).total_seconds() >= 86400:
            self.trades_this_day = 0
            self.day_reset_time = now
        
        # Check limits
        if self.trades_this_hour >= self.config.max_trades_per_hour:
            return False, f"Hourly limit reached ({self.trades_this_hour}/{self.config.max_trades_per_hour})"
        
        if self.trades_this_day >= self.config.max_trades_per_day:
            return False, f"Daily limit reached ({self.trades_this_day}/{self.config.max_trades_per_day})"
        
        return True, "ok"
    
    def _increment_trade_count(self) -> None:
        """Increment trade counters"""
        self.trades_this_hour += 1
        self.trades_this_day += 1
        logger.debug(f"Trades: hour={self.trades_this_hour}, day={self.trades_this_day}")
    
    # ============================================================
    # Trading Logic
    # ============================================================
    
    async def process_signal(self, signal: Signal) -> None:
        """
        🔥 Process AI signal and execute trading logic.
        
        Main entry point for signal processing.
        """
        self.last_signal = signal
        self.last_decision_time = datetime.now(timezone.utc)
        
        # Update market regime
        self.market_regime = self._detect_market_regime(signal)
        logger.info(f"Market Regime: {self.market_regime.value}")
        
        # State machine logic
        if self.state == TradingState.PAUSED:
            logger.debug("Engine paused, ignoring signal")
            return
        
        if self.state == TradingState.ERROR:
            logger.warning("Engine in error state, skipping signal")
            return
        
        # Check cooldown
        now = datetime.now(timezone.utc)
        if self.cooldown_until and now < self.cooldown_until:
            remaining = (self.cooldown_until - now).total_seconds()
            logger.info(f"In cooldown, {remaining:.0f}s remaining")
            return
        
        # Check rate limits
        can_trade, limit_reason = self._check_rate_limits()
        if not can_trade:
            logger.warning(f"Rate limit: {limit_reason}")
            return
        
        # Get regime parameters
        regime_params = self._get_regime_params()
        
        # Check signal validity
        direction: Direction = signal["direction"]
        confidence: float = signal["confidence"]
        price: Optional[float] = signal.get("price")
        
        if direction == "flat":
            # Check if we should close existing position
            if self.state == TradingState.IN_POSITION:
                await self._close_position(reason="signal_flat")
            else:
                self._change_state(TradingState.IDLE, "flat_signal")
            return
        
        if confidence < regime_params["min_confidence"]:
            logger.info(
                f"Confidence {confidence:.2%} < {regime_params['min_confidence']:.2%} "
                f"for {self.market_regime.value}"
            )
            return
        
        if price is None or price <= 0:
            logger.warning("Invalid price in signal")
            return
        
        # Calculate position parameters
        await self._execute_trade(signal, regime_params)
    
    async def _execute_trade(
        self,
        signal: Signal,
        regime_params: Dict[str, float],
    ) -> None:
        """Execute trade based on signal and regime parameters"""
        
        direction: Direction = signal["direction"]
        confidence: float = signal["confidence"]
        price: float = signal["price"]
        reason: str = signal.get("reason", "")
        
        # Calculate SL (1% default)
        sl_pct = 0.01 * regime_params["sl_multiplier"]
        if direction == "long":
            sl_price = price * (1.0 - sl_pct)
        else:
            sl_price = price * (1.0 + sl_pct)
        
        # Get volatility for risk assessment
        volatility = None
        scores = signal.get("scores", {})
        if "volatility" in scores:
            volatility = scores["volatility"]
        
        # Risk assessment
        if self.config.require_risk_approval:
            risk_assessment = self.risk_manager.assess_trade(
                symbol=self.ai_brain.symbol,
                direction=direction,
                entry_price=price,
                sl_price=sl_price,
                tp_price=None,  # Will be calculated by risk manager
                confidence=confidence,
                volatility=volatility,
            )
            
            if not risk_assessment.approved:
                logger.warning(
                    f"Trade rejected by risk manager: {risk_assessment.reason} | "
                    f"Warnings: {risk_assessment.warnings}"
                )
                return
            
            # Use risk-adjusted size
            position_size = risk_assessment.position_size * regime_params["size_multiplier"]
            tp_price = risk_assessment.tp_price
            
            # Apply regime TP adjustment
            if tp_price and price:
                tp_distance = abs(tp_price - price)
                adjusted_tp_distance = tp_distance * regime_params["tp_multiplier"]
                if direction == "long":
                    tp_price = price + adjusted_tp_distance
                else:
                    tp_price = price - adjusted_tp_distance
            
            sl_price = risk_assessment.sl_price  # Use risk manager SL
            
            logger.info(
                f"Risk approved: size={position_size:.4f}, TP={tp_price:.2f}, "
                f"SL={sl_price:.2f}, R:R={risk_assessment.rr_ratio:.2f}"
            )
        else:
            # Simple sizing without risk manager
            position_size = 0.01 * regime_params["size_multiplier"]
            tp_distance = sl_pct * 2.0 * regime_params["tp_multiplier"]
            if direction == "long":
                tp_price = price * (1.0 + tp_distance)
            else:
                tp_price = price * (1.0 - tp_distance)
        
        # Execute trade via executor
        try:
            self._change_state(TradingState.PENDING_ENTRY, f"Opening {direction}")
            
            await self.trade_executor.handle_signal({
                "dir": direction,
                "conf": confidence,
                "price": price,
                "reason": reason,
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            
            # Register in risk manager
            self.risk_manager.register_position(
                symbol=self.ai_brain.symbol,
                direction=direction,
                size=position_size,
                entry_price=price,
            )
            
            self._change_state(TradingState.IN_POSITION, f"{direction} opened")
            self._increment_trade_count()
            
            # Callback
            if self.on_trade_opened:
                try:
                    self.on_trade_opened(signal, position_size, tp_price, sl_price)
                except Exception as e:
                    logger.error(f"Error in on_trade_opened callback: {e}")
        
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}", exc_info=True)
            self._change_state(TradingState.ERROR, str(e))
    
    async def _close_position(self, reason: str = "") -> None:
        """Close current position"""
        
        if self.state != TradingState.IN_POSITION:
            logger.debug("No position to close")
            return
        
        self._change_state(TradingState.PENDING_EXIT, reason)
        
        try:
            # Get position from manager
            position = self.position_manager.get_position(self.ai_brain.symbol)
            
            if position:
                # Close via executor (sends market order)
                await self.trade_executor._close_position()
                
                # Determine if win/loss
                is_win = position.unrealized_pnl > 0
                
                # Update risk manager
                close_price = position.entry_price  # Fallback, should be updated
                if self.last_signal and self.last_signal.get("price"):
                    close_price = self.last_signal["price"]
                
                self.risk_manager.close_position(
                    symbol=self.ai_brain.symbol,
                    exit_price=close_price,
                    is_win=is_win,
                )
                
                # Enter cooldown if loss
                if not is_win and self.config.cooldown_after_loss_seconds > 0:
                    self.cooldown_until = datetime.now(timezone.utc)
                    self.cooldown_until = self.cooldown_until.replace(
                        second=self.cooldown_until.second + self.config.cooldown_after_loss_seconds
                    )
                    self._change_state(TradingState.COOLDOWN, f"Loss cooldown {self.config.cooldown_after_loss_seconds}s")
                    logger.warning(f"Entering cooldown until {self.cooldown_until}")
                else:
                    self._change_state(TradingState.IDLE, "Position closed")
                
                # Callback
                if self.on_trade_closed:
                    try:
                        self.on_trade_closed(position, is_win)
                    except Exception as e:
                        logger.error(f"Error in on_trade_closed callback: {e}")
            else:
                logger.warning("Position not found in manager")
                self._change_state(TradingState.IDLE, "Position not found")
        
        except Exception as e:
            logger.error(f"Failed to close position: {e}", exc_info=True)
            self._change_state(TradingState.ERROR, str(e))
    
    # ============================================================
    # Background Loop
    # ============================================================
    
    async def start(self) -> None:
        """Start trading engine"""
        self._running = True
        
        # Start dependencies
        await self.trade_executor.connect()
        await self.position_manager.start()
        
        # Start main loop
        self._main_loop_task = asyncio.create_task(self._main_loop())
        
        logger.info("🔥 Trading Engine started")
    
    async def stop(self) -> None:
        """Stop trading engine"""
        self._running = False
        
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Stop dependencies
        await self.position_manager.stop()
        await self.trade_executor.close()
        
        logger.info("Trading Engine stopped")
    
    async def _main_loop(self) -> None:
        """Main trading loop - gets signals from AI Brain and processes them"""
        
        while self._running:
            try:
                # Get latest signal from AI Brain
                signal = self.ai_brain.build_signal()
                
                # Process signal
                await self.process_signal(signal)
                
                # Update position manager with current price
                if signal.get("price"):
                    current_price = signal["price"]
                    action = self.position_manager.update_position(
                        symbol=self.ai_brain.symbol,
                        current_price=current_price,
                    )
                    
                    # If TP/SL hit, close position
                    if action and action.get("action") == "close":
                        await self._close_position(reason=action.get("reason", "tp_sl_hit"))
                
                # Log stats periodically
                if self.trades_this_day % 10 == 0 and self.trades_this_day > 0:
                    self.risk_manager.log_stats()
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                self._change_state(TradingState.ERROR, str(e))
            
            await asyncio.sleep(2.0)  # Run every 2 seconds
    
    # ============================================================
    # Manual Controls
    # ============================================================
    
    def pause(self) -> None:
        """Pause trading engine"""
        self._change_state(TradingState.PAUSED, "Manual pause")
        logger.warning("Trading Engine PAUSED")
    
    def resume(self) -> None:
        """Resume trading engine"""
        if self.state == TradingState.PAUSED:
            self._change_state(TradingState.IDLE, "Manual resume")
            logger.info("Trading Engine RESUMED")
    
    async def emergency_close_all(self) -> None:
        """Emergency close all positions"""
        logger.warning("EMERGENCY CLOSE ALL POSITIONS")
        await self._close_position(reason="emergency")
    
    # ============================================================
    # Reporting
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            "state": self.state.value,
            "market_regime": self.market_regime.value,
            "running": self._running,
            "in_cooldown": self.cooldown_until is not None and datetime.now(timezone.utc) < self.cooldown_until,
            "cooldown_remaining": (
                (self.cooldown_until - datetime.now(timezone.utc)).total_seconds()
                if self.cooldown_until and datetime.now(timezone.utc) < self.cooldown_until
                else 0
            ),
            "trades_this_hour": self.trades_this_hour,
            "trades_this_day": self.trades_this_day,
            "last_decision": self.last_decision_time.isoformat() if self.last_decision_time else None,
            "risk_stats": self.risk_manager.get_stats(),
            "position_stats": self.position_manager.get_pnl_summary(),
        }
