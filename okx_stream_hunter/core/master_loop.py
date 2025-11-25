"""
Master Trading Loop - PHASE 3
Real-time tick processing, candle reconstruction, multi-signal fusion
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from okx_stream_hunter.core.market_state import MarketState
from okx_stream_hunter.ai.brain_ultra import get_brain
from okx_stream_hunter.core.trade_supervisor import TradeSupervisor
from okx_stream_hunter.core.ai_safety import AISafetyLayer
from okx_stream_hunter.core.circuit_breaker import CircuitBreaker
from okx_stream_hunter.integrations.risk_manager import RiskManager
from okx_stream_hunter.integrations.position_manager import PositionManager
from okx_stream_hunter.integrations.execution_engine import ExecutionEngine

logger = logging.getLogger(__name__)


class CandleBuilder:
    """Real-time candle reconstruction from ticks"""
    
    def __init__(self, timeframe_seconds: int = 60):
        self.timeframe = timeframe_seconds
        self.current_candle = None
        self.completed_candles = deque(maxlen=500)
        self.tick_buffer = deque(maxlen=1000)
    
    def process_tick(self, price: float, volume: float, timestamp: datetime) -> Optional[Dict]:
        """Process tick and build candles"""
        self.tick_buffer.append({'price': price, 'volume': volume, 'timestamp': timestamp})
        
        candle_start = self._get_candle_start(timestamp)
        
        if self.current_candle is None:
            self.current_candle = self._init_candle(price, volume, candle_start)
            return None
        
        if timestamp >= self.current_candle['close_time']:
            completed = self.current_candle.copy()
            self.completed_candles.append(completed)
            
            self.current_candle = self._init_candle(price, volume, candle_start)
            
            return completed
        
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += volume
        self.current_candle['tick_count'] += 1
        
        return None
    
    def _get_candle_start(self, timestamp: datetime) -> datetime:
        """Get candle period start time"""
        seconds = timestamp.timestamp()
        candle_seconds = (seconds // self.timeframe) * self.timeframe
        return datetime.fromtimestamp(candle_seconds)
    
    def _init_candle(self, price: float, volume: float, start_time: datetime) -> Dict:
        """Initialize new candle"""
        return {
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
            'open_time': start_time,
            'close_time': start_time + timedelta(seconds=self.timeframe),
            'tick_count': 1
        }
    
    def get_recent_candles(self, count: int = 50) -> list:
        """Get recent completed candles"""
        return list(self.completed_candles)[-count:]


class MasterTradingLoop:
    """Master trading loop with full system integration"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        self.candle_builder = CandleBuilder(timeframe_seconds=config.get('candle_timeframe', 60))
        
        self.ai_brain = get_brain()
        self.trade_supervisor = TradeSupervisor(config)
        self.ai_safety = AISafetyLayer(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager()
        self.execution_engine = ExecutionEngine(config=config)
        
        self.decision_interval = config.get('decision_interval_seconds', 5)
        self.last_decision_time = None
        
        self.market_state = None
        self.last_ai_decision = None
        
        self.stats = {
            'ticks_processed': 0,
            'candles_completed': 0,
            'decisions_made': 0,
            'trades_executed': 0,
            'safety_blocks': 0,
            'circuit_trips': 0
        }
        
        logger.info("🚀 Master Trading Loop initialized")
    
    async def start(self):
        """Start master loop"""
        self.running = True
        logger.info("=" * 70)
        logger.info("🔥 MASTER TRADING LOOP STARTED - PHASE 3 ACTIVE")
        logger.info("=" * 70)
        
        decision_task = asyncio.create_task(self._decision_loop())
        supervision_task = asyncio.create_task(self._supervision_loop())
        
        try:
            await asyncio.gather(decision_task, supervision_task)
        except Exception as e:
            logger.error(f"❌ Master loop error: {e}")
        finally:
            self.running = False
    
    def stop(self):
        """Stop master loop"""
        self.running = False
        logger.info("⏹️ Master Trading Loop stopped")
    
    async def process_tick(self, tick_data: Dict):
        """Process incoming tick data"""
        try:
            price = tick_data.get('price', 0.0)
            volume = tick_data.get('size', 0.0)
            timestamp = tick_data.get('timestamp', datetime.now())
            
            self.stats['ticks_processed'] += 1
            
            completed_candle = self.candle_builder.process_tick(price, volume, timestamp)
            
            if completed_candle:
                self.stats['candles_completed'] += 1
                await self._on_candle_complete(completed_candle)
        
        except Exception as e:
            logger.error(f"❌ Tick processing error: {e}")
    
    async def update_market_state(self, market_state: MarketState):
        """Update current market state"""
        self.market_state = market_state
        
        self.ai_brain.on_ticker({
            'last': market_state.price,
            'bidPx': market_state.bid,
            'askPx': market_state.ask,
            'vol24h': market_state.volume_24h
        })
    
    async def _on_candle_complete(self, candle: Dict):
        """Handle completed candle"""
        self.ai_brain.on_candle(candle)
        
        logger.debug(
            f"📊 Candle: O={candle['open']:.2f} H={candle['high']:.2f} "
            f"L={candle['low']:.2f} C={candle['close']:.2f} V={candle['volume']:.2f}"
        )
    
    async def _decision_loop(self):
        """Main decision-making loop"""
        while self.running:
            try:
                await asyncio.sleep(self.decision_interval)
                
                if not self._should_make_decision():
                    continue
                
                await self._make_trading_decision()
                
            except Exception as e:
                logger.error(f"❌ Decision loop error: {e}")
                await asyncio.sleep(self.decision_interval)
    
    async def _supervision_loop(self):
        """Trade supervision loop"""
        while self.running:
            try:
                await asyncio.sleep(1)
                
                if self.market_state:
                    await self._supervise_active_trades()
                
            except Exception as e:
                logger.error(f"❌ Supervision loop error: {e}")
                await asyncio.sleep(1)
    
    def _should_make_decision(self) -> bool:
        """Check if should make new decision"""
        if self.last_decision_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_decision_time).total_seconds()
        return elapsed >= self.decision_interval
    
    async def _make_trading_decision(self):
        """Execute full trading decision pipeline"""
        try:
            self.last_decision_time = datetime.now()
            self.stats['decisions_made'] += 1
            
            circuit_check = await self.circuit_breaker.check(before_trade=True)
            if not circuit_check['allowed']:
                self.stats['circuit_trips'] += 1
                logger.warning(f"⚡ Circuit: {circuit_check['reason']}")
                return
            
            if not self.market_state:
                logger.warning("⚠️ No market state available")
                return
            
            ai_decision = self.ai_brain.get_live_decision()
            
            if not ai_decision:
                logger.debug("💤 AI: No decision (warming up)")
                return
            
            self.last_ai_decision = ai_decision
            
            safety_check = await self.ai_safety.validate_decision(ai_decision)
            if not safety_check['safe']:
                self.stats['safety_blocks'] += 1
                logger.warning(f"🛡️ Safety blocked: {safety_check['blocks']}")
                return
            
            supervisor_validation = await self.trade_supervisor.validate_decision(
                ai_decision,
                self.market_state.to_dict()
            )
            
            if not supervisor_validation['approved']:
                logger.info(f"👮 Supervisor blocked: {supervisor_validation['reason']}")
                return
            
            if ai_decision['direction'] == 'NEUTRAL':
                logger.debug("😴 AI Decision: NEUTRAL (no action)")
                return
            
            logger.info(
                f"🎯 AI Decision: {ai_decision['direction']} @ {ai_decision['confidence']:.2%} "
                f"| {ai_decision['reason']}"
            )
            
            if not self.config.get('auto_trading', False):
                logger.info("⏸️ Auto-trading disabled - decision logged only")
                return
            
            risk_approved = await self.risk_manager.check_trade_approval(
                ai_decision,
                self.market_state.to_dict()
            )
            
            if not risk_approved['approved']:
                logger.warning(f"🚫 Risk rejected: {risk_approved.get('reason')}")
                return
            
            position_size = await self.risk_manager.calculate_position_size(
                ai_decision,
                self.market_state.to_dict()
            )
            
            if position_size == 0:
                logger.warning("⚠️ Position size = 0, skipping trade")
                return
            
            execution_result = await self.execution_engine.execute_signal(
                ai_decision,
                position_size,
                self.market_state.to_dict()
            )
            
            if execution_result:
                self.stats['trades_executed'] += 1
                
                self.trade_supervisor.register_trade({
                    'trade_id': execution_result.get('trade_id'),
                    'entry_price': self.market_state.price,
                    'direction': ai_decision['direction'],
                    'size': position_size,
                    'sl': ai_decision.get('sl'),
                    'tp': ai_decision.get('tp'),
                    'confidence': ai_decision['confidence'],
                    'regime': ai_decision.get('regime', 'unknown')
                })
                
                logger.info(f"✅ Trade executed: {execution_result.get('trade_id')}")
        
        except Exception as e:
            logger.error(f"❌ Decision pipeline error: {e}", exc_info=True)
    
    async def _supervise_active_trades(self):
        """Supervise all active trades"""
        try:
            active_trades = self.trade_supervisor.get_active_trades_status()
            
            for trade_status in active_trades:
                action = await self.trade_supervisor.supervise_trade(
                    trade_status['trade_id'],
                    self.market_state.to_dict()
                )
                
                if action and action['type'] == 'close':
                    await self._close_trade(trade_status['trade_id'], action['reason'])
        
        except Exception as e:
            logger.error(f"❌ Supervision error: {e}")
    
    async def _close_trade(self, trade_id: str, reason: str):
        """Close trade"""
        try:
            close_result = await self.execution_engine.close_position(
                trade_id,
                self.market_state.price,
                reason
            )
            
            if close_result:
                await self.circuit_breaker.record_trade(close_result)
                
                self.ai_safety.record_trade_outcome(close_result.get('pnl', 0.0))
                
                self.trade_supervisor.unregister_trade(trade_id, close_result)
                
                logger.info(
                    f"🔒 Trade closed: {trade_id} | "
                    f"PnL: ${close_result.get('pnl', 0):.2f} ({close_result.get('pnl_pct', 0):.2%}) | "
                    f"Reason: {reason}"
                )
        
        except Exception as e:
            logger.error(f"❌ Close trade error: {e}")
    
    def get_stats(self) -> Dict:
        """Get loop statistics"""
        return {
            **self.stats,
            'running': self.running,
            'last_decision': self.last_decision_time.isoformat() if self.last_decision_time else None,
            'candles_in_buffer': len(self.candle_builder.completed_candles),
            'active_trades': len(self.trade_supervisor.active_trades),
            'circuit_breaker': self.circuit_breaker.get_status(),
            'ai_safety': self.ai_safety.get_safety_status(),
            'supervisor': self.trade_supervisor.get_stats()
        }
