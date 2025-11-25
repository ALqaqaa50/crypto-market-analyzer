"""
PHASE 3: Autonomous Trading Runtime
Complete system integration with all safety layers
"""

import asyncio
import logging
import signal
from typing import Dict, Optional
from datetime import datetime

from okx_stream_hunter.core.stream_engine import OKXStreamEngine
from okx_stream_hunter.core.market_state import MarketState
from okx_stream_hunter.core.master_loop import MasterTradingLoop
from okx_stream_hunter.core.watchdog import SystemWatchdog
from okx_stream_hunter.core.adaptive_limiter import AdaptiveRateLimiter
from okx_stream_hunter.ai.brain_ultra import get_brain
from okx_stream_hunter.integrations.risk_manager import RiskManager
from okx_stream_hunter.integrations.position_manager import PositionManager

logger = logging.getLogger(__name__)


class AutonomousRuntime:
    """
    PHASE 3: Autonomous Trading Runtime
    
    Integrates all components:
    - Stream Engine (WebSocket data ingestion)
    - Master Trading Loop (tick→candle→decision→execution)
    - AI Brain (PROMETHEUS v7)
    - Safety Layers (AI Safety, Circuit Breaker, Trade Supervisor)
    - System Watchdog (health monitoring & auto-recovery)
    - Adaptive Rate Limiter (API protection)
    - Risk Manager & Position Manager
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Core components
        self.stream_engine: Optional[OKXStreamEngine] = None
        self.master_loop: Optional[MasterTradingLoop] = None
        self.watchdog: Optional[SystemWatchdog] = None
        self.rate_limiter: Optional[AdaptiveRateLimiter] = None
        
        # AI Brain
        self.ai_brain = None
        
        # State
        self.market_state = MarketState(symbol=config.get('symbol', 'BTC-USDT-SWAP'))
        
        # Statistics
        self.start_time = None
        self.stats = {
            'uptime_seconds': 0,
            'total_ticks': 0,
            'total_decisions': 0,
            'total_trades': 0,
            'total_recoveries': 0,
            'component_health': {}
        }
        
        logger.info("🚀 Autonomous Runtime initialized")
    
    async def start(self):
        """Start autonomous trading system"""
        if self.running:
            logger.warning("Runtime already running")
            return
        
        self.running = True
        self.start_time = datetime.now()
        
        logger.info("=" * 80)
        logger.info("🔥 PHASE 3: AUTONOMOUS TRADING SYSTEM STARTING")
        logger.info("=" * 80)
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Register signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Start all components
            await self._start_components()
            
            logger.info("✅ All components started successfully")
            logger.info("🤖 Autonomous Trading System is now LIVE")
            logger.info("=" * 80)
            
            # Main runtime loop
            await self._runtime_loop()
            
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def _initialize_components(self):
        """Initialize all system components"""
        logger.info("🔧 Initializing components...")
        
        # Rate Limiter
        self.rate_limiter = AdaptiveRateLimiter({
            'base_requests_per_second': self.config.get('rate_limit_base', 10),
            'min_requests_per_second': self.config.get('rate_limit_min', 1),
            'max_requests_per_second': self.config.get('rate_limit_max', 50)
        })
        logger.info("✅ Rate Limiter initialized")
        
        # Master Trading Loop
        self.master_loop = MasterTradingLoop(self.config)
        logger.info("✅ Master Trading Loop initialized")
        
        # Stream Engine
        symbol = self.config.get('symbol', 'BTC-USDT-SWAP')
        self.stream_engine = OKXStreamEngine(symbol=symbol)
        
        # Subscribe to stream events
        self.stream_engine.subscribe('ticker', self._on_ticker)
        self.stream_engine.subscribe('trades', self._on_trade)
        self.stream_engine.subscribe('orderbook', self._on_orderbook)
        
        logger.info(f"✅ Stream Engine initialized for {symbol}")
        
        # Initialize AI Brain
        from okx_stream_hunter.ai.brain_ultra import get_brain
        self.ai_brain = get_brain()
        logger.info("✅ AI Brain initialized and ready for data feed")
        
        # System Watchdog
        watchdog_config = {
            'heartbeat_interval_seconds': self.config.get('watchdog_interval', 10),
            'component_timeout_seconds': 30,
            'failure_threshold': self.config.get('watchdog_failure_threshold', 3)
        }
        self.watchdog = SystemWatchdog(watchdog_config)
        
        # Register components with watchdog
        self.watchdog.register_component(
            'stream_engine',
            health_check=self._check_stream_health,
            recovery_callback=self._recover_stream
        )
        
        self.watchdog.register_component(
            'master_loop',
            health_check=self._check_master_loop_health,
            recovery_callback=self._recover_master_loop
        )
        
        self.watchdog.register_component(
            'rate_limiter',
            health_check=self._check_rate_limiter_health
        )
        
        logger.info("✅ System Watchdog initialized with 3 components")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logger.info(f"⚠️ Received signal {sig}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())
        
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            logger.warning(f"Could not setup signal handlers: {e}")
    
    async def _start_components(self):
        """Start all components"""
        logger.info("▶️ Starting components...")
        
        # Start Master Loop
        asyncio.create_task(self.master_loop.start())
        logger.info("✅ Master Trading Loop started")
        
        # Start Stream Engine
        asyncio.create_task(self.stream_engine.start())
        logger.info("✅ Stream Engine started")
        
        # Start Watchdog
        asyncio.create_task(self.watchdog.start())
        logger.info("✅ System Watchdog started")
        
        # Give components time to initialize
        await asyncio.sleep(2)
    
    async def _runtime_loop(self):
        """Main runtime monitoring loop"""
        logger.info("🔄 Runtime loop active")
        
        stats_interval = self.config.get('stats_interval', 60)
        last_stats_time = datetime.now()
        
        while self.running and not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(1)
                
                # Update uptime
                if self.start_time:
                    self.stats['uptime_seconds'] = (datetime.now() - self.start_time).total_seconds()
                
                # Periodic stats logging
                if (datetime.now() - last_stats_time).total_seconds() >= stats_interval:
                    self._log_stats()
                    last_stats_time = datetime.now()
                
            except Exception as e:
                logger.error(f"❌ Runtime loop error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop autonomous trading system gracefully"""
        if not self.running:
            return
        
        logger.info("=" * 80)
        logger.info("🛑 AUTONOMOUS TRADING SYSTEM SHUTDOWN INITIATED")
        logger.info("=" * 80)
        
        self.running = False
        self.shutdown_event.set()
        
        # Stop Master Loop first (closes positions)
        if self.master_loop:
            logger.info("⏹️ Stopping Master Trading Loop...")
            self.master_loop.stop()
            await asyncio.sleep(2)
        
        # Stop Stream Engine
        if self.stream_engine:
            logger.info("⏹️ Stopping Stream Engine...")
            await self.stream_engine.stop()
        
        # Stop Watchdog
        if self.watchdog:
            logger.info("⏹️ Stopping System Watchdog...")
            await self.watchdog.stop()
        
        # Final stats
        self._log_stats()
        
        logger.info("=" * 80)
        logger.info("✅ AUTONOMOUS TRADING SYSTEM SHUTDOWN COMPLETE")
        logger.info("=" * 80)
    
    # ============================================================
    # Stream Callbacks
    # ============================================================
    
    async def _on_ticker(self, ticker_data: Dict):
        """Handle ticker updates"""
        try:
            self.market_state.price = ticker_data.get('last', 0.0)
            self.market_state.bid = ticker_data.get('bidPx', 0.0)
            self.market_state.ask = ticker_data.get('askPx', 0.0)
            self.market_state.volume_24h = ticker_data.get('vol24h', 0.0)
            
            # Feed AI Brain (CRITICAL: This was missing!)
            if self.ai_brain and hasattr(self.ai_brain, 'on_ticker'):
                try:
                    self.ai_brain.on_ticker(ticker_data)
                except Exception as e:
                    logger.error(f"AI Brain ticker update failed: {e}")
            
            # Update master loop
            if self.master_loop:
                await self.master_loop.update_market_state(self.market_state)
            
        except Exception as e:
            logger.error(f"❌ Ticker callback error: {e}")
    
    async def _on_trade(self, trade_data: Dict):
        """Handle trade updates"""
        try:
            self.stats['total_ticks'] += 1
            
            price = float(trade_data.get('px', 0))
            size = float(trade_data.get('sz', 0))
            timestamp = datetime.now()
            
            # Feed AI Brain (CRITICAL: This was missing!)
            if self.ai_brain and hasattr(self.ai_brain, 'on_trade'):
                try:
                    self.ai_brain.on_trade(trade_data)
                except Exception as e:
                    logger.error(f"AI Brain trade update failed: {e}")
            
            # Process tick in master loop
            if self.master_loop:
                await self.master_loop.process_tick({
                    'price': price,
                    'size': size,
                    'timestamp': timestamp
                })
            
            # Update market state
            self.market_state.price = price
            
        except Exception as e:
            logger.error(f"❌ Trade callback error: {e}")
    
    async def _on_orderbook(self, orderbook_data: Dict):
        """Handle orderbook updates"""
        try:
            bids = orderbook_data.get('bids', [])
            asks = orderbook_data.get('asks', [])
            
            if bids:
                self.market_state.bid = float(bids[0][0])
            if asks:
                self.market_state.ask = float(asks[0][0])
            
            # Feed AI Brain (CRITICAL: This was missing!)
            if self.ai_brain and hasattr(self.ai_brain, 'on_orderbook'):
                try:
                    self.ai_brain.on_orderbook(orderbook_data)
                except Exception as e:
                    logger.error(f"AI Brain orderbook update failed: {e}")
            
        except Exception as e:
            logger.error(f"❌ Orderbook callback error: {e}")
    
    # ============================================================
    # Health Checks & Recovery
    # ============================================================
    
    async def _check_stream_health(self) -> bool:
        """Check stream engine health"""
        if not self.stream_engine:
            return False
        
        # OKXStreamEngine uses 'running' flag and 'ws' attribute
        if hasattr(self.stream_engine, 'running'):
            is_running = self.stream_engine.running
            has_connection = (
                hasattr(self.stream_engine, 'ws') and 
                self.stream_engine.ws is not None and
                not self.stream_engine.ws.closed
            )
            return is_running and has_connection
        
        return True
    
    async def _check_master_loop_health(self) -> bool:
        """Check master loop health"""
        if not self.master_loop:
            return False
        
        return self.master_loop.running
    
    async def _check_rate_limiter_health(self) -> bool:
        """Check rate limiter health"""
        if not self.rate_limiter:
            return False
        
        stats = self.rate_limiter.get_stats()
        return stats.get('current_limit', 0) > 0
    
    async def _recover_stream(self):
        """Recover stream engine"""
        logger.warning("🔄 Attempting to recover Stream Engine...")
        self.stats['total_recoveries'] += 1
        
        try:
            if self.stream_engine:
                await self.stream_engine.stop()
                await asyncio.sleep(2)
                asyncio.create_task(self.stream_engine.start())
                logger.info("✅ Stream Engine recovery initiated")
        except Exception as e:
            logger.error(f"❌ Stream recovery failed: {e}")
    
    async def _recover_master_loop(self):
        """Recover master loop"""
        logger.warning("🔄 Attempting to recover Master Loop...")
        self.stats['total_recoveries'] += 1
        
        try:
            if self.master_loop:
                self.master_loop.stop()
                await asyncio.sleep(2)
                asyncio.create_task(self.master_loop.start())
                logger.info("✅ Master Loop recovery initiated")
        except Exception as e:
            logger.error(f"❌ Master Loop recovery failed: {e}")
    
    # ============================================================
    # Statistics
    # ============================================================
    
    def _log_stats(self):
        """Log system statistics"""
        if not self.master_loop:
            return
        
        loop_stats = self.master_loop.get_stats()
        watchdog_health = self.watchdog.get_overall_health() if self.watchdog else {}
        
        logger.info("=" * 80)
        logger.info("📊 AUTONOMOUS RUNTIME STATISTICS")
        logger.info(f"⏱️  Uptime: {self.stats['uptime_seconds']:.0f}s")
        logger.info(f"📡 Ticks Processed: {self.stats['total_ticks']}")
        logger.info(f"🎯 AI Decisions: {loop_stats.get('decisions_made', 0)}")
        logger.info(f"💼 Trades Executed: {loop_stats.get('trades_executed', 0)}")
        logger.info(f"🛡️ Safety Blocks: {loop_stats.get('safety_blocks', 0)}")
        logger.info(f"⚡ Circuit Trips: {loop_stats.get('circuit_trips', 0)}")
        logger.info(f"🔄 Auto Recoveries: {self.stats['total_recoveries']}")
        logger.info(f"💚 System Health: {watchdog_health.get('overall_status', 'unknown')}")
        logger.info("=" * 80)
    
    def get_status(self) -> Dict:
        """Get runtime status"""
        loop_stats = self.master_loop.get_stats() if self.master_loop else {}
        watchdog_health = self.watchdog.get_overall_health() if self.watchdog else {}
        
        return {
            'running': self.running,
            'uptime_seconds': self.stats['uptime_seconds'],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stats': self.stats,
            'master_loop': loop_stats,
            'watchdog': watchdog_health,
            'market_state': self.market_state.to_dict()
        }


# ============================================================
# Global Instance Management
# ============================================================

_runtime_instance: Optional[AutonomousRuntime] = None


def get_runtime() -> Optional[AutonomousRuntime]:
    """Get global runtime instance"""
    return _runtime_instance


def set_runtime(runtime: AutonomousRuntime):
    """Set global runtime instance"""
    global _runtime_instance
    _runtime_instance = runtime


async def start_autonomous_runtime(config: Dict):
    """Start autonomous runtime"""
    runtime = AutonomousRuntime(config)
    set_runtime(runtime)
    await runtime.start()


async def stop_autonomous_runtime():
    """Stop autonomous runtime"""
    runtime = get_runtime()
    if runtime:
        await runtime.stop()
