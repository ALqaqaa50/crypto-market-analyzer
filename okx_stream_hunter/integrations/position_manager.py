# okx_stream_hunter/integrations/position_manager.py
"""
🔥 Position Manager - Advanced Position Tracking with TP/SL Management
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Any, Callable

logger = logging.getLogger("position.manager")

Direction = Literal["long", "short", "flat"]


class PositionManager:
    """Enhanced Position Manager for Phase 2"""
    
    def __init__(self):
        self.positions = {}
        logger.info("📊 Position Manager initialized")
    
    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        regime: str,
        max_risk_pct: float = 0.02
    ) -> float:
        """Calculate appropriate position size"""
        try:
            if stop_loss == 0 or entry_price == 0:
                logger.warning("Invalid SL or entry price")
                return 0.0
            
            risk_per_unit = abs(entry_price - stop_loss)
            
            if risk_per_unit == 0:
                logger.warning("Zero risk - SL too close to entry")
                return 0.0
            
            max_risk_amount = balance * max_risk_pct
            
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            regime_multipliers = {
                'trend_up': 1.2,
                'trend_down': 1.2,
                'breakout': 1.3,
                'range': 0.8,
                'volatility_expansion': 0.6,
                'unknown': 0.7
            }
            regime_multiplier = regime_multipliers.get(regime, 1.0)
            
            adjusted_risk = max_risk_amount * confidence_multiplier * regime_multiplier
            
            position_size = adjusted_risk / risk_per_unit
            
            max_position = balance * 0.1 / entry_price
            position_size = min(position_size, max_position)
            
            position_size = max(0.001, min(1.0, position_size))
            
            logger.info(f"📐 Position Size: {position_size:.4f} | Risk: ${adjusted_risk:.2f} | Confidence: {confidence:.2%}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Position size calculation error: {e}")
            return 0.0


logger = logging.getLogger("position.manager")

Direction = Literal["long", "short", "flat"]


@dataclass
class Position:
    """Single position with TP/SL tracking"""
    
    symbol: str
    direction: Direction
    size: float
    entry_price: float
    opened_at: datetime
    
    # TP/SL levels
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    
    # Trailing stop
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.015  # 1.5%
    highest_price: Optional[float] = None  # for long
    lowest_price: Optional[float] = None   # for short
    
    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Metadata
    position_id: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
    
    # Lifecycle
    closed_at: Optional[datetime] = None
    close_reason: str = ""


@dataclass
class PositionManagerConfig:
    """Configuration for position manager"""
    
    # TP/SL defaults
    default_tp_pct: float = 0.02      # 2% TP
    default_sl_pct: float = 0.01      # 1% SL
    default_rr_ratio: float = 2.0     # Risk:Reward = 1:2
    
    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_activation_pct: float = 0.01   # Start trailing after 1% profit
    trailing_distance_pct: float = 0.005    # Trail 0.5% behind peak
    
    # Risk management
    max_position_age_minutes: int = 120     # Auto-close after 2 hours
    max_drawdown_pct: float = 0.05          # Close if -5% from entry
    
    # Break-even
    enable_breakeven_move: bool = True
    breakeven_trigger_pct: float = 0.015    # Move SL to BE after 1.5% profit
    breakeven_offset_pct: float = 0.002     # Place BE slightly above entry
    
    # Auto-sizing (NEW)
    enable_auto_sizing: bool = True
    base_position_size: float = 0.01        # Base size in BTC
    confidence_multiplier: float = 2.0      # Max multiplier for high confidence
    
    # Dynamic leverage (NEW)
    enable_dynamic_leverage: bool = True
    min_leverage: float = 1.0
    max_leverage: float = 5.0
    leverage_confidence_curve: float = 2.0  # Exponential scaling


class PositionManager:
    """
    🔥 Advanced Position Manager
    
    Features:
    - Track multiple positions (currently supporting 1 active position per symbol)
    - Automatic TP/SL calculation based on R:R ratio
    - Trailing stop loss for winning trades
    - Break-even SL adjustment
    - Time-based exits
    - P&L tracking
    - Event callbacks for position lifecycle
    """
    
    def __init__(
        self,
        config: Optional[PositionManagerConfig] = None,
        on_tp_hit: Optional[Callable] = None,
        on_sl_hit: Optional[Callable] = None,
        on_position_closed: Optional[Callable] = None,
    ):
        self.config = config or PositionManagerConfig()
        
        # Active positions by symbol
        self.positions: Dict[str, Position] = {}
        
        # Closed positions history
        self.closed_positions: List[Position] = []
        
        # Callbacks
        self.on_tp_hit = on_tp_hit
        self.on_sl_hit = on_sl_hit
        self.on_position_closed = on_position_closed
        
        self._running = False
        self._watchdog_task: Optional[asyncio.Task] = None
    
    # ============================================================
    # Auto-Sizing & Dynamic Leverage (NEW)
    # ============================================================
    
    def calculate_optimal_size(
        self,
        base_size: float,
        confidence: float,
        volatility: float = 0.0,
    ) -> float:
        """
        Calculate optimal position size based on confidence
        
        Formula: size = base_size * (1 + confidence * (multiplier - 1))
        
        Higher confidence = larger size (up to multiplier cap)
        Higher volatility = smaller size
        """
        if not self.config.enable_auto_sizing:
            return base_size
        
        # Confidence scaling (0.0 - 1.0 -> 1.0x - multiplier)
        size_multiplier = 1.0 + confidence * (self.config.confidence_multiplier - 1.0)
        
        # Volatility adjustment (reduce size in high volatility)
        if volatility > 0.02:  # High volatility (>2%)
            vol_adjustment = 0.7
        elif volatility > 0.015:
            vol_adjustment = 0.85
        else:
            vol_adjustment = 1.0
        
        optimal_size = base_size * size_multiplier * vol_adjustment
        
        logger.info(
            f"Auto-sizing: base={base_size}, confidence={confidence:.2f}, "
            f"multiplier={size_multiplier:.2f}, vol_adj={vol_adjustment:.2f}, "
            f"final={optimal_size:.4f}"
        )
        
        return optimal_size
    
    def calculate_dynamic_leverage(
        self,
        confidence: float,
        volatility: float = 0.0,
    ) -> float:
        """
        Calculate dynamic leverage based on signal confidence
        
        Formula: leverage = min_lev + (max_lev - min_lev) * confidence^curve
        
        Exponential curve ensures conservative leverage for low confidence
        """
        if not self.config.enable_dynamic_leverage:
            return self.config.min_leverage
        
        # Exponential scaling (confidence^2 by default)
        leverage_factor = confidence ** self.config.leverage_confidence_curve
        
        leverage = (
            self.config.min_leverage +
            (self.config.max_leverage - self.config.min_leverage) * leverage_factor
        )
        
        # Volatility reduction (lower leverage in volatile markets)
        if volatility > 0.02:
            leverage *= 0.6
        elif volatility > 0.015:
            leverage *= 0.8
        
        leverage = max(self.config.min_leverage, min(self.config.max_leverage, leverage))
        
        logger.info(
            f"Dynamic leverage: confidence={confidence:.2f}, factor={leverage_factor:.2f}, "
            f"volatility={volatility:.4f}, final={leverage:.2f}x"
        )
        
        return leverage
    
    # ============================================================
    # Position Lifecycle
    # ============================================================
    
    def open_position(
        self,
        symbol: str,
        direction: Direction,
        size: float,
        entry_price: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        reason: str = "",
        confidence: float = 0.0,
    ) -> Position:
        """
        Open a new position with TP/SL levels.
        
        If TP/SL not provided, they will be calculated based on config.
        """
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}, closing old one first")
            self.close_position(symbol, reason="replaced")
        
        # Calculate TP/SL if not provided
        if tp_price is None or sl_price is None:
            tp_price, sl_price = self._calculate_tp_sl(direction, entry_price)
        
        position = Position(
            symbol=symbol,
            direction=direction,
            size=size,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            opened_at=datetime.now(timezone.utc),
            reason=reason,
            confidence=confidence,
            trailing_stop_enabled=self.config.enable_trailing_stop,
            trailing_stop_pct=self.config.trailing_distance_pct,
            highest_price=entry_price if direction == "long" else None,
            lowest_price=entry_price if direction == "short" else None,
            position_id=f"{symbol}_{int(time.time() * 1000)}",
        )
        
        self.positions[symbol] = position
        
        logger.info(
            f"✅ Position OPENED: {symbol} {direction.upper()} size={size:.4f} "
            f"entry={entry_price:.2f} TP={tp_price:.2f} SL={sl_price:.2f} "
            f"reason={reason} conf={confidence:.2%}"
        )
        
        return position
    
    def close_position(
        self,
        symbol: str,
        close_price: Optional[float] = None,
        reason: str = "",
    ) -> Optional[Position]:
        """Close an active position and calculate P&L"""
        
        if symbol not in self.positions:
            logger.warning(f"No active position for {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # Use last known price if not provided
        if close_price is None:
            close_price = position.entry_price
        
        # Calculate realized P&L
        if position.direction == "long":
            pnl_pct = (close_price - position.entry_price) / position.entry_price
        elif position.direction == "short":
            pnl_pct = (position.entry_price - close_price) / position.entry_price
        else:
            pnl_pct = 0.0
        
        position.realized_pnl = pnl_pct * position.size
        position.closed_at = datetime.now(timezone.utc)
        position.close_reason = reason
        
        # Move to history
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        logger.info(
            f"❌ Position CLOSED: {symbol} {position.direction.upper()} "
            f"entry={position.entry_price:.2f} close={close_price:.2f} "
            f"PnL={pnl_pct:.2%} reason={reason}"
        )
        
        # Trigger callback
        if self.on_position_closed:
            try:
                self.on_position_closed(position, close_price)
            except Exception as e:
                logger.error(f"Error in on_position_closed callback: {e}")
        
        return position
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Update position with current price and check TP/SL/Trailing.
        
        Returns action dict if TP/SL hit, None otherwise.
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Update unrealized P&L
        if position.direction == "long":
            pnl_pct = (current_price - position.entry_price) / position.entry_price
        elif position.direction == "short":
            pnl_pct = (position.entry_price - current_price) / position.entry_price
        else:
            pnl_pct = 0.0
        
        position.unrealized_pnl = pnl_pct * position.size
        
        # Check TP hit
        if position.tp_price:
            if position.direction == "long" and current_price >= position.tp_price:
                logger.info(f"🎯 TP HIT for {symbol} @ {current_price:.2f}")
                if self.on_tp_hit:
                    try:
                        self.on_tp_hit(position, current_price)
                    except Exception as e:
                        logger.error(f"Error in on_tp_hit callback: {e}")
                
                self.close_position(symbol, current_price, reason="tp_hit")
                return {"action": "close", "reason": "tp_hit", "price": current_price}
            
            if position.direction == "short" and current_price <= position.tp_price:
                logger.info(f"🎯 TP HIT for {symbol} @ {current_price:.2f}")
                if self.on_tp_hit:
                    try:
                        self.on_tp_hit(position, current_price)
                    except Exception as e:
                        logger.error(f"Error in on_tp_hit callback: {e}")
                
                self.close_position(symbol, current_price, reason="tp_hit")
                return {"action": "close", "reason": "tp_hit", "price": current_price}
        
        # Update trailing stop
        if position.trailing_stop_enabled:
            self._update_trailing_stop(position, current_price)
        
        # Check SL hit (including trailing SL)
        if position.sl_price:
            if position.direction == "long" and current_price <= position.sl_price:
                logger.info(f"🛑 SL HIT for {symbol} @ {current_price:.2f}")
                if self.on_sl_hit:
                    try:
                        self.on_sl_hit(position, current_price)
                    except Exception as e:
                        logger.error(f"Error in on_sl_hit callback: {e}")
                
                self.close_position(symbol, current_price, reason="sl_hit")
                return {"action": "close", "reason": "sl_hit", "price": current_price}
            
            if position.direction == "short" and current_price >= position.sl_price:
                logger.info(f"🛑 SL HIT for {symbol} @ {current_price:.2f}")
                if self.on_sl_hit:
                    try:
                        self.on_sl_hit(position, current_price)
                    except Exception as e:
                        logger.error(f"Error in on_sl_hit callback: {e}")
                
                self.close_position(symbol, current_price, reason="sl_hit")
                return {"action": "close", "reason": "sl_hit", "price": current_price}
        
        # Check break-even move
        if self.config.enable_breakeven_move:
            self._check_breakeven_move(position, current_price)
        
        return None
    
    # ============================================================
    # TP/SL Management
    # ============================================================
    
    def _calculate_tp_sl(
        self,
        direction: Direction,
        entry_price: float,
    ) -> tuple[float, float]:
        """Calculate TP and SL based on R:R ratio and config"""
        
        sl_pct = self.config.default_sl_pct
        tp_pct = sl_pct * self.config.default_rr_ratio
        
        if direction == "long":
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)
        elif direction == "short":
            sl_price = entry_price * (1.0 + sl_pct)
            tp_price = entry_price * (1.0 - tp_pct)
        else:
            sl_price = entry_price
            tp_price = entry_price
        
        return tp_price, sl_price
    
    def _update_trailing_stop(self, position: Position, current_price: float) -> None:
        """Update trailing stop based on peak price"""
        
        if position.direction == "long":
            # Update highest price
            if position.highest_price is None or current_price > position.highest_price:
                position.highest_price = current_price
            
            # Check if we should activate trailing
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct >= self.config.trailing_activation_pct:
                # Calculate new trailing SL
                new_sl = position.highest_price * (1.0 - position.trailing_stop_pct)
                
                # Only move SL up, never down
                if position.sl_price is None or new_sl > position.sl_price:
                    old_sl = position.sl_price
                    position.sl_price = new_sl
                    logger.debug(
                        f"Trailing SL updated for {position.symbol}: "
                        f"{old_sl:.2f} -> {new_sl:.2f} (peak={position.highest_price:.2f})"
                    )
        
        elif position.direction == "short":
            # Update lowest price
            if position.lowest_price is None or current_price < position.lowest_price:
                position.lowest_price = current_price
            
            # Check if we should activate trailing
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct >= self.config.trailing_activation_pct:
                # Calculate new trailing SL
                new_sl = position.lowest_price * (1.0 + position.trailing_stop_pct)
                
                # Only move SL down, never up
                if position.sl_price is None or new_sl < position.sl_price:
                    old_sl = position.sl_price
                    position.sl_price = new_sl
                    logger.debug(
                        f"Trailing SL updated for {position.symbol}: "
                        f"{old_sl:.2f} -> {new_sl:.2f} (low={position.lowest_price:.2f})"
                    )
    
    def _check_breakeven_move(self, position: Position, current_price: float) -> None:
        """Move SL to break-even after sufficient profit"""
        
        if position.direction == "long":
            profit_pct = (current_price - position.entry_price) / position.entry_price
            if profit_pct >= self.config.breakeven_trigger_pct:
                be_price = position.entry_price * (1.0 + self.config.breakeven_offset_pct)
                if position.sl_price is None or position.sl_price < be_price:
                    position.sl_price = be_price
                    logger.info(
                        f"✅ Break-even SL set for {position.symbol} @ {be_price:.2f}"
                    )
        
        elif position.direction == "short":
            profit_pct = (position.entry_price - current_price) / position.entry_price
            if profit_pct >= self.config.breakeven_trigger_pct:
                be_price = position.entry_price * (1.0 - self.config.breakeven_offset_pct)
                if position.sl_price is None or position.sl_price > be_price:
                    position.sl_price = be_price
                    logger.info(
                        f"✅ Break-even SL set for {position.symbol} @ {be_price:.2f}"
                    )
    
    # ============================================================
    # Queries & Stats
    # ============================================================
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get active position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all active positions"""
        return self.positions.copy()
    
    def get_closed_positions(self, limit: int = 100) -> List[Position]:
        """Get recent closed positions"""
        return self.closed_positions[-limit:]
    
    def get_pnl_summary(self) -> Dict[str, Any]:
        """Calculate total P&L and stats"""
        
        total_realized = sum(p.realized_pnl for p in self.closed_positions)
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        
        wins = [p for p in self.closed_positions if p.realized_pnl > 0]
        losses = [p for p in self.closed_positions if p.realized_pnl < 0]
        
        win_rate = len(wins) / len(self.closed_positions) if self.closed_positions else 0.0
        
        return {
            "total_realized_pnl": total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl": total_realized + total_unrealized,
            "closed_trades": len(self.closed_positions),
            "active_positions": len(self.positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
        }
    
    # ============================================================
    # Background Watchdog
    # ============================================================
    
    async def start(self) -> None:
        """Start position manager watchdog"""
        self._running = True
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Position Manager started")
    
    async def stop(self) -> None:
        """Stop position manager"""
        self._running = False
        if self._watchdog_task:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
        logger.info("Position Manager stopped")
    
    async def _watchdog_loop(self) -> None:
        """Periodic checks for time-based exits and max drawdown"""
        
        while self._running:
            try:
                await self._check_time_based_exits()
                await self._check_max_drawdown()
            except Exception as e:
                logger.error(f"Error in position manager watchdog: {e}", exc_info=True)
            
            await asyncio.sleep(10.0)  # Check every 10 seconds
    
    async def _check_time_based_exits(self) -> None:
        """Close positions that are too old"""
        
        now = datetime.now(timezone.utc)
        
        for symbol, position in list(self.positions.items()):
            age_minutes = (now - position.opened_at).total_seconds() / 60.0
            
            if age_minutes >= self.config.max_position_age_minutes:
                logger.info(
                    f"⏰ Closing {symbol} due to max age: {age_minutes:.1f} min"
                )
                self.close_position(symbol, reason="max_age")
    
    async def _check_max_drawdown(self) -> None:
        """Close positions with excessive drawdown"""
        
        for symbol, position in list(self.positions.items()):
            if position.unrealized_pnl < 0:
                loss_pct = abs(position.unrealized_pnl / position.size)
                
                if loss_pct >= self.config.max_drawdown_pct:
                    logger.info(
                        f"⚠️  Closing {symbol} due to max drawdown: {loss_pct:.2%}"
                    )
                    self.close_position(symbol, reason="max_drawdown")
