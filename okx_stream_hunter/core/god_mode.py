# okx_stream_hunter/core/god_mode.py
"""
🔥👑 GOD MODE - Ultimate AI Trading System Orchestrator (v3.0 ULTRA)

The most advanced trading mode combining:
- Real-time AI Brain analysis (7 detection layers)
- Automatic trading execution
- Advanced risk management
- Position management with auto-sizing
- Self-learning & pattern recognition
- Hyperparameter optimization
- System stability & recovery
- Live market integration
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from .ai_brain import AIBrain, Signal
from .trading_engine import TradingEngine, TradingEngineConfig, TradingState
from .stability import StabilityManager, RecoveryConfig
from ..ai.learning import SelfLearningEngine
from ..integrations.risk_manager import RiskManager, RiskConfig
from ..integrations.position_manager import PositionManager, PositionManagerConfig
from ..integrations.trade_executor import TradeExecutor

logger = logging.getLogger("god.mode")


class GodMode:
    """
    🔥👑 GOD MODE - Ultimate Trading System (v3.0 ULTRA)
    
    New Features (v3.0):
    ✅ 7-layer AI detection (orderflow, liquidity, spoof, regime, microstructure)
    ✅ ATR-based dynamic TP/SL
    ✅ Advanced strategy detection (trend/range/breakout/reversal)
    ✅ Auto position sizing with dynamic leverage
    ✅ Self-learning engine with pattern recognition
    ✅ Hyperparameter auto-optimization
    ✅ System stability with crash recovery
    ✅ Heartbeat monitoring and log rotation
    
    Core Features:
    - Full AI-powered trading automation
    - Real-time market analysis
    - Advanced risk management
    - Dynamic position sizing
    - Market regime adaptation
    - Performance tracking & self-learning
    - Emergency controls
    - Comprehensive logging
    """
    
    def __init__(
        self,
        symbol: str,
        initial_balance: float = 1000.0,
        risk_per_trade: float = 0.01,  # 1%
        max_daily_loss: float = 0.05,  # 5%
        enable_live_trading: bool = False,  # Safety: paper trading by default
        enable_learning: bool = True,  # NEW: Enable self-learning
        enable_stability: bool = True,  # NEW: Enable stability manager
    ):
        """
        Initialize GOD MODE (v3.0 ULTRA).
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT-SWAP")
            initial_balance: Starting capital
            risk_per_trade: Risk per trade as % of capital
            max_daily_loss: Maximum daily loss %
            enable_live_trading: Enable REAL trading (use with caution!)
            enable_learning: Enable self-learning engine
            enable_stability: Enable stability manager
        """
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.enable_live_trading = enable_live_trading
        self.enable_learning = enable_learning
        self.enable_stability = enable_stability
        
        logger.info("🔥👑 Initializing GOD MODE v3.0 ULTRA...")
        
        # Initialize AI Brain (ENHANCED)
        self.ai_brain = AIBrain(symbol=symbol, logger=logger)
        logger.info("✅ AI Brain initialized with 7 detection layers")
        
        # Initialize Risk Manager (ENHANCED)
        risk_config = RiskConfig(
            account_balance=initial_balance,
            max_risk_per_trade_pct=risk_per_trade,
            max_daily_loss_pct=max_daily_loss,
        )
        self.risk_manager = RiskManager(config=risk_config)
        logger.info("✅ Risk Manager initialized with smart TP/SL calculator")
        
        # Initialize Position Manager (ENHANCED)
        position_config = PositionManagerConfig(
            enable_auto_sizing=True,  # NEW
            enable_dynamic_leverage=True,  # NEW
        )
        self.position_manager = PositionManager(
            config=position_config,
            on_tp_hit=self._on_tp_hit,
            on_sl_hit=self._on_sl_hit,
            on_position_closed=self._on_position_closed,
        )
        logger.info("✅ Position Manager initialized with auto-sizing & dynamic leverage")
        
        # Initialize Trade Executor (conditional)
        if enable_live_trading:
            logger.warning("🚨 LIVE TRADING ENABLED - REAL MONEY AT RISK 🚨")
            self.trade_executor = TradeExecutor(position_manager=self.position_manager)
        else:
            logger.info("📄 Paper Trading Mode - No real trades will be executed")
            self.trade_executor = None  # Paper trading
        
        # Initialize Trading Engine
        trading_config = TradingEngineConfig(
            adapt_to_regime=True,
            enable_state_machine=True,
        )
        self.trading_engine = TradingEngine(
            ai_brain=self.ai_brain,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
            trade_executor=self.trade_executor,
            config=trading_config,
            on_state_change=self._on_state_change,
            on_trade_opened=self._on_trade_opened,
            on_trade_closed=self._on_trade_closed_engine,
        )
        logger.info("✅ Trading Engine initialized with regime adaptation")
        
        # Initialize Self-Learning Engine (NEW)
        self.learning_engine: Optional[SelfLearningEngine] = None
        if enable_learning:
            self.learning_engine = SelfLearningEngine()
            logger.info("✅ Self-Learning Engine initialized (pattern recognition + auto-tuning)")
        
        # Initialize Stability Manager (NEW)
        self.stability_manager: Optional[StabilityManager] = None
        if enable_stability:
            recovery_config = RecoveryConfig(
                enable_auto_recovery=True,
                heartbeat_interval_seconds=30,
                enable_log_rotation=True,
            )
            self.stability_manager = StabilityManager(
                config=recovery_config,
                on_recovery=self._on_system_recovery,
                on_heartbeat_timeout=self._on_heartbeat_timeout,
            )
            logger.info("✅ Stability Manager initialized (recovery + heartbeat + log rotation)")
        
        # State
        self._running = False
        self._mode = "paper" if not enable_live_trading else "LIVE"
        
        # Performance tracking
        self.session_start_time: Optional[datetime] = None
        self.total_signals: int = 0
        self.total_trades: int = 0
        
        # Learning data
        self.learning_data: List[Dict] = []
        
        logger.info(f"🔥👑 GOD MODE v3.0 ULTRA initialized successfully ({self._mode} mode)")
        
        logger.info("="*60)
        logger.info("👑 GOD MODE INITIALIZED 👑")
        logger.info("="*60)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Mode: {self._mode}")
        logger.info(f"Initial Balance: ${initial_balance:.2f}")
        logger.info(f"Risk per Trade: {risk_per_trade:.1%}")
        logger.info(f"Max Daily Loss: {max_daily_loss:.1%}")
        logger.info("="*60)
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    async def start(self) -> None:
        """🔥 Start GOD MODE v3.0 ULTRA"""
        
        if self._running:
            logger.warning("GOD MODE already running")
            return
        
        self._running = True
        self.session_start_time = datetime.now(timezone.utc)
        
        logger.info("="*60)
        logger.info("🚀 STARTING GOD MODE v3.0 ULTRA")
        logger.info("="*60)
        
        # Start Stability Manager (NEW)
        if self.stability_manager:
            await self.stability_manager.start()
        
        # Start Self-Learning Engine (NEW)
        if self.learning_engine:
            await self.learning_engine.start()
        
        # Start core components
        await self.ai_brain.start()
        await self.position_manager.start()
        await self.trading_engine.start()
        
        logger.info("✅ All systems operational")
        logger.info("="*60)
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
        
        logger.info("✅ GOD MODE v3.0 ULTRA ACTIVE - All Systems Online")
        logger.info("="*60)
    
    async def stop(self) -> None:
        """Stop GOD MODE gracefully"""
        
        if not self._running:
            return
        
        logger.info("="*60)
        logger.info("🛑 STOPPING GOD MODE v3.0 ULTRA")
        logger.info("="*60)
        
        self._running = False
        
        # Stop trading engine
        await self.trading_engine.stop()
        
        # Stop position manager
        await self.position_manager.stop()
        
        # Stop AI brain
        await self.ai_brain.stop()
        
        # Stop learning engine (NEW)
        if self.learning_engine:
            await self.learning_engine.stop()
        
        # Stop stability manager (NEW)
        if self.stability_manager:
            await self.stability_manager.stop()
        
        # Export session data
        self._export_session_summary()
        
        logger.info("✅ GOD MODE v3.0 ULTRA STOPPED")
        logger.info("="*60)
        logger.info("="*60)
    
    async def emergency_stop(self) -> None:
        """🚨 Emergency stop - close all positions immediately"""
        
        logger.critical("="*60)
        logger.critical("🚨 EMERGENCY STOP ACTIVATED 🚨")
        logger.critical("="*60)
        
        # Close all positions
        await self.trading_engine.emergency_close_all()
        
        # Pause engine
        self.trading_engine.pause()
        
        # Stop
        await self.stop()
        
        logger.critical("🚨 EMERGENCY STOP COMPLETE 🚨")
    
    # ============================================================
    # Monitoring & Learning
    # ============================================================
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop - logs stats and learns from performance"""
        
        while self._running:
            try:
                # Log stats every 5 minutes
                await asyncio.sleep(300)
                
                self._log_status()
                
                # Self-learning: analyze recent performance
                self._analyze_and_learn()
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
    
    def _log_status(self) -> None:
        """Log current status"""
        
        engine_status = self.trading_engine.get_status()
        risk_stats = self.risk_manager.get_stats()
        position_stats = self.position_manager.get_pnl_summary()
        
        logger.info("="*60)
        logger.info("👑 GOD MODE STATUS")
        logger.info("="*60)
        logger.info(f"State: {engine_status['state']}")
        logger.info(f"Market Regime: {engine_status['market_regime']}")
        logger.info(f"Balance: ${risk_stats['account_balance']:.2f}")
        logger.info(f"Daily P&L: ${risk_stats['daily_pnl']:.2f} ({risk_stats['daily_pnl_pct']:.2%})")
        logger.info(f"Total P&L: ${risk_stats['total_pnl']:.2f}")
        logger.info(f"Win Rate: {risk_stats['win_rate']:.1%}")
        logger.info(f"Open Positions: {position_stats['active_positions']}")
        logger.info(f"Total Trades: {risk_stats['total_trades']}")
        logger.info("="*60)
    
    def _analyze_and_learn(self) -> None:
        """
        🧠 Self-learning: analyze recent performance and adapt (ENHANCED v3.0).
        
        New Features:
        - Pattern recognition from historical trades
        - Hyperparameter optimization suggestions
        - Adaptive strategy adjustment
        """
        
        risk_stats = self.risk_manager.get_stats()
        
        # Collect learning data point
        learning_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_rate": risk_stats['win_rate'],
            "total_pnl": risk_stats['total_pnl'],
            "daily_pnl": risk_stats['daily_pnl'],
            "consecutive_wins": risk_stats['consecutive_wins'],
            "consecutive_losses": risk_stats['consecutive_losses'],
            "market_regime": self.trading_engine.market_regime.value,
            "state": self.trading_engine.state.value,
        }
        
        self.learning_data.append(learning_entry)
        
        # Learning Engine Integration (NEW)
        if self.learning_engine:
            # Get best patterns
            best_patterns = self.learning_engine.get_best_patterns(top_n=5)
            if best_patterns:
                logger.info(
                    f"🧠 Top Pattern: {best_patterns[0].pattern_type}, "
                    f"Win Rate: {best_patterns[0].win_rate:.1%}, "
                    f"Confidence: {best_patterns[0].confidence:.2f}"
                )
            
            # Get best hyperparameters
            best_hp = self.learning_engine.get_best_hyperparameters()
            if best_hp:
                logger.info(
                    f"⚙️ Best Hyperparameters: {best_hp.name}, "
                    f"Score: {best_hp.score:.3f}, "
                    f"Win Rate: {best_hp.win_rate:.1%}"
                )
        
        # Adaptive logic: adjust parameters based on performance
        if risk_stats['consecutive_losses'] >= 3:
            logger.warning("⚠️  Adaptive: Multiple losses detected - system will reduce exposure")
            # Risk manager automatically handles this via loss reduction factor
        
        if risk_stats['win_rate'] > 0.7 and risk_stats['total_trades'] > 10:
            logger.info("✅ High win rate detected - system performing well")
        
        # Stability Manager Integration (NEW)
        if self.stability_manager:
            # Update heartbeat
            self.stability_manager.heartbeat()
            
            # Record signal generation
            self.stability_manager.record_signal()
        
        # Export learning data periodically
        if len(self.learning_data) % 20 == 0:
            self._export_learning_data()
    
    # ============================================================
    # Callbacks
    # ============================================================
    
    def _on_state_change(self, old_state: TradingState, new_state: TradingState, reason: str) -> None:
        """Callback when trading engine changes state"""
        logger.info(f"🔄 State Change: {old_state.value} → {new_state.value} ({reason})")
    
    def _on_trade_opened(self, signal: Signal, size: float, tp: float, sl: float) -> None:
        """Callback when trade is opened"""
        self.total_trades += 1
        logger.info(
            f"📈 Trade Opened: {signal['direction'].upper()} | "
            f"Size: {size:.4f} | TP: {tp:.2f} | SL: {sl:.2f} | "
            f"Confidence: {signal['confidence']:.1%}"
        )
    
    def _on_trade_closed_engine(self, position, is_win: bool) -> None:
        """Callback when trade is closed by engine"""
        result = "WIN ✅" if is_win else "LOSS ❌"
        logger.info(
            f"📉 Trade Closed: {result} | "
            f"Entry: {position.entry_price:.2f} | "
            f"P&L: ${position.realized_pnl:.2f} | "
            f"Reason: {position.close_reason}"
        )
    
    def _on_tp_hit(self, position, price: float) -> None:
        """Callback when TP is hit"""
        logger.info(f"🎯 TP HIT: {position.symbol} @ ${price:.2f} - PROFIT SECURED")
    
    def _on_sl_hit(self, position, price: float) -> None:
        """Callback when SL is hit"""
        logger.warning(f"🛑 SL HIT: {position.symbol} @ ${price:.2f} - LOSS CUT")
    
    def _on_position_closed(self, position, price: float) -> None:
        """Callback when position is closed by position manager"""
        # Record trade for learning (NEW)
        if self.learning_engine and position.closed_at:
            duration = (position.closed_at - position.opened_at).total_seconds() / 60
            
            signal_data = {
                "direction": position.direction,
                "confidence": position.confidence,
                "reason": position.reason,
            }
            
            self.learning_engine.record_trade(
                signal_data=signal_data,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=position.realized_pnl,
                duration_minutes=int(duration),
                success=position.realized_pnl > 0,
            )
    
    def _on_system_recovery(self, prev_state) -> None:
        """Callback when system recovers from crash (NEW)"""
        logger.warning(
            f"🔧 System recovered from previous crash! "
            f"Previous uptime: {prev_state.uptime_seconds/60:.1f}m, "
            f"Open positions: {prev_state.open_positions}"
        )
        
        # TODO: Restore open positions if needed
        # For now, start fresh
    
    def _on_heartbeat_timeout(self) -> None:
        """Callback when heartbeat times out (NEW)"""
        logger.critical("💔 HEARTBEAT TIMEOUT - System may be hanging!")
        
        # Emergency: pause trading
        self.trading_engine.pause()
    
    # ============================================================
    # Manual Controls
    # ============================================================
    
    def pause(self) -> None:
        """Pause trading (stop taking new positions)"""
        self.trading_engine.pause()
        logger.warning("⏸️  GOD MODE PAUSED - No new positions will be opened")
    
    def resume(self) -> None:
        """Resume trading"""
        self.trading_engine.resume()
        logger.info("▶️  GOD MODE RESUMED - Trading active")
    
    async def close_all_positions(self) -> None:
        """Close all open positions"""
        logger.warning("Closing all positions...")
        await self.trading_engine.emergency_close_all()
    
    # ============================================================
    # Data Export
    # ============================================================
    
    def _export_session_summary(self) -> None:
        """Export session summary to file"""
        
        summary = {
            "session_start": self.session_start_time.isoformat() if self.session_start_time else None,
            "session_end": datetime.now(timezone.utc).isoformat(),
            "mode": self._mode,
            "symbol": self.symbol,
            "initial_balance": self.initial_balance,
            "risk_stats": self.risk_manager.get_stats(),
            "position_stats": self.position_manager.get_pnl_summary(),
            "engine_status": self.trading_engine.get_status(),
        }
        
        try:
            filename = f"god_mode_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Session summary exported to {filename}")
        except Exception as e:
            logger.error(f"Failed to export session summary: {e}")
    
    def _export_learning_data(self) -> None:
        """Export learning data for future analysis/training"""
        
        try:
            with open("god_mode_learning_data.json", "w") as f:
                json.dump(self.learning_data, f, indent=2)
            logger.debug(f"Learning data exported ({len(self.learning_data)} entries)")
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")
    
    # ============================================================
    # Reporting
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GOD MODE status"""
        
        return {
            "mode": self._mode,
            "running": self._running,
            "symbol": self.symbol,
            "session_start": self.session_start_time.isoformat() if self.session_start_time else None,
            "engine": self.trading_engine.get_status(),
            "risk": self.risk_manager.get_stats(),
            "positions": self.position_manager.get_pnl_summary(),
            "total_signals": self.total_signals,
            "total_trades": self.total_trades,
        }
    
    def print_status(self) -> None:
        """Print detailed status to console"""
        self._log_status()


# ============================================================
# Helper function to launch GOD MODE
# ============================================================

async def launch_god_mode(
    symbol: str = "BTC-USDT-SWAP",
    initial_balance: float = 1000.0,
    enable_live_trading: bool = False,
) -> GodMode:
    """
    🔥👑 Launch GOD MODE
    
    Quick start function for GOD MODE.
    
    Args:
        symbol: Trading symbol
        initial_balance: Starting capital
        enable_live_trading: Enable REAL trading (⚠️  DANGER!)
    
    Returns:
        GodMode instance (already started)
    
    Example:
        ```python
        # Paper trading
        god = await launch_god_mode()
        
        # Live trading (use with extreme caution!)
        god = await launch_god_mode(enable_live_trading=True)
        
        # Stop when done
        await god.stop()
        ```
    """
    god = GodMode(
        symbol=symbol,
        initial_balance=initial_balance,
        enable_live_trading=enable_live_trading,
    )
    
    await god.start()
    
    return god
