"""
Trading Orchestrator - Main coordinator for live trading system
Integrates Stream Engine + AI Brain + Risk Management + Execution
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

from okx_stream_hunter.core.stream_engine import OKXStreamEngine
from okx_stream_hunter.core.market_state import MarketState
from okx_stream_hunter.ai.brain_ultra import get_brain
from okx_stream_hunter.ai.rl_agent import RLAgent
from okx_stream_hunter.integrations.risk_manager import RiskManager
from okx_stream_hunter.integrations.position_manager import PositionManager
from okx_stream_hunter.integrations.execution_engine import ExecutionEngine, ExecutionMode

logger = logging.getLogger(__name__)


class TradingOrchestrator:
    """Main orchestrator for automated trading system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.running = False
        
        symbol = config.get('symbol', 'BTC-USDT-SWAP')
        paper_trading = config.get('paper_trading', True)
        self.auto_trading_enabled = config.get('auto_trading', False)
        
        self.stream_engine = OKXStreamEngine(symbol=symbol)
        
        self.ai_brain = get_brain()
        
        self.rl_agent = RLAgent(config)
        self.rl_agent.load_state()
        
        self.risk_manager = RiskManager(config)
        self.position_manager = PositionManager()
        
        mode = ExecutionMode.PAPER if paper_trading else ExecutionMode.LIVE
        self.execution_engine = ExecutionEngine(mode=mode, config=config)
        
        self.last_decision = None
        self.last_decision_time = None
        self.decision_interval = config.get('decision_interval_seconds', 5)
        
        self.state = {
            'status': 'initializing',
            'ai_active': True,
            'auto_trading': self.auto_trading_enabled,
            'paper_trading': paper_trading,
            'health': 'healthy'
        }
        
        logger.info("🎯 Trading Orchestrator initialized")
        logger.info(f"   Symbol: {symbol}")
        logger.info(f"   Paper Trading: {paper_trading}")
        logger.info(f"   Auto Trading: {self.auto_trading_enabled}")
    
    async def start(self):
        """Start the trading system"""
        try:
            logger.info("=" * 60)
            logger.info("🚀 PROMETHEUS v7 TRADING SYSTEM STARTING")
            logger.info("=" * 60)
            
            self.stream_engine.subscribe('state_update', self._on_market_state_update)
            self.stream_engine.subscribe('ticker', self._on_ticker)
            self.stream_engine.subscribe('trades', self._on_trades)
            self.stream_engine.subscribe('orderbook', self._on_orderbook)
            
            self.running = True
            self.state['status'] = 'running'
            
            stream_task = asyncio.create_task(self.stream_engine.start())
            
            trading_task = asyncio.create_task(self._trading_loop())
            
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("✅ All systems online")
            logger.info("=" * 60)
            
            await asyncio.gather(stream_task, trading_task, monitoring_task)
            
        except Exception as e:
            logger.error(f"❌ Orchestrator error: {e}")
            await self.stop()
    
    async def _on_market_state_update(self, market_state: MarketState):
        """Handle market state updates"""
        try:
            candle_data = {
                'open': market_state.price,
                'high': market_state.price,
                'low': market_state.price,
                'close': market_state.price,
                'volume': market_state.volume_window,
                'timestamp': market_state.timestamp
            }
            self.ai_brain.on_candle(candle_data)
            
        except Exception as e:
            logger.error(f"Market state update error: {e}")
    
    async def _on_ticker(self, ticker: Dict):
        """Handle ticker updates"""
        try:
            self.ai_brain.on_ticker(ticker)
        except Exception as e:
            logger.error(f"Ticker update error: {e}")
    
    async def _on_trades(self, trades: list):
        """Handle trades updates"""
        try:
            self.ai_brain.on_trades(trades)
        except Exception as e:
            logger.error(f"Trades update error: {e}")
    
    async def _on_orderbook(self, orderbook: Dict):
        """Handle orderbook updates"""
        try:
            self.ai_brain.on_orderbook(orderbook)
        except Exception as e:
            logger.error(f"Orderbook update error: {e}")
    
    async def _trading_loop(self):
        """Main trading decision loop"""
        logger.info("🔄 Trading loop started")
        
        while self.running:
            try:
                await asyncio.sleep(self.decision_interval)
                
                if not self.auto_trading_enabled:
                    continue
                
                market_state = self.stream_engine.get_latest_market_state()
                
                if market_state.price == 0:
                    continue
                
                await self._check_existing_positions(market_state)
                
                decision = self.ai_brain.get_live_decision()
                
                if decision is None:
                    continue
                
                self.last_decision = decision
                self.last_decision_time = datetime.now()
                
                regime = decision.get('regime', 'unknown')
                
                should_trade = self.rl_agent.should_trade(decision, regime)
                
                if not should_trade:
                    logger.info("🚫 RL Agent rejected trade")
                    continue
                
                if not self.risk_manager.can_trade():
                    logger.warning("🚫 Risk Manager blocked trade")
                    self.state['health'] = 'risk_locked'
                    continue
                
                if self.execution_engine.open_positions:
                    logger.info("⚠️ Position already open, skipping")
                    continue
                
                direction = decision.get('direction', 'NEUTRAL')
                
                if direction == 'NEUTRAL':
                    continue
                
                position_size = self.position_manager.calculate_position_size(
                    balance=self.execution_engine.paper_balance,
                    entry_price=market_state.price,
                    stop_loss=decision.get('sl', 0),
                    confidence=decision.get('confidence', 0),
                    regime=regime
                )
                
                if position_size <= 0:
                    logger.warning("❌ Invalid position size calculated")
                    continue
                
                logger.info(f"🎯 EXECUTING TRADE: {direction} {position_size} @ {market_state.price:.2f}")
                
                result = await self.execution_engine.execute_signal(
                    decision=decision,
                    position_size=position_size,
                    market_state=market_state.to_dict()
                )
                
                if result:
                    self.risk_manager.register_trade(result)
                    logger.info("✅ Trade executed successfully")
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_existing_positions(self, market_state: MarketState):
        """Check SL/TP for existing positions"""
        try:
            result = await self.execution_engine.check_stop_loss_take_profit(market_state.price)
            
            if result and 'closed_positions' in result:
                for closed_pos in result['closed_positions']:
                    self.rl_agent.update_from_trade({
                        'pnl': closed_pos.get('pnl', 0),
                        'pattern': closed_pos.get('pattern', 'unknown'),
                        'regime': closed_pos.get('regime', 'unknown'),
                        'confidence': closed_pos.get('confidence', 0),
                        'signal': closed_pos.get('signal', 'UNKNOWN')
                    })
                    
                    logger.info("📊 RL Agent updated from closed trade")
                
                self.rl_agent.save_state()
                self.execution_engine.save_trades_log()
        
        except Exception as e:
            logger.error(f"Position check error: {e}")
    
    async def _monitoring_loop(self):
        """System monitoring and health checks"""
        logger.info("💓 Monitoring loop started")
        
        while self.running:
            try:
                await asyncio.sleep(60)
                
                stats = self.get_system_stats()
                
                logger.info("=" * 60)
                logger.info("📊 SYSTEM STATUS")
                logger.info(f"   Balance: ${stats['execution']['balance']:,.2f}")
                logger.info(f"   Open Positions: {stats['execution']['open_positions']}")
                logger.info(f"   Total Trades: {stats['execution']['closed_trades']}")
                logger.info(f"   Win Rate: {stats['execution']['win_rate']:.2%}")
                logger.info(f"   Total PnL: ${stats['execution']['total_pnl']:,.2f}")
                logger.info(f"   RL Confidence Threshold: {stats['rl']['adaptive_parameters']['min_confidence_threshold']:.2f}")
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        return {
            'state': self.state.copy(),
            'market': self.stream_engine.get_latest_market_state().to_dict(),
            'ai': self.ai_brain.get_status(),
            'rl': self.rl_agent.get_stats(),
            'execution': self.execution_engine.get_stats(),
            'risk': self.risk_manager.get_stats(),
            'last_decision': self.last_decision,
            'last_decision_time': self.last_decision_time.isoformat() if self.last_decision_time else None
        }
    
    def enable_auto_trading(self):
        """Enable auto trading"""
        self.auto_trading_enabled = True
        self.state['auto_trading'] = True
        logger.info("✅ Auto Trading ENABLED")
    
    def disable_auto_trading(self):
        """Disable auto trading"""
        self.auto_trading_enabled = False
        self.state['auto_trading'] = False
        logger.info("⏸️ Auto Trading DISABLED")
    
    async def stop(self):
        """Stop the trading system"""
        logger.info("⏹️ Stopping trading system...")
        
        self.running = False
        self.state['status'] = 'stopped'
        
        await self.stream_engine.stop()
        
        self.rl_agent.save_state()
        self.execution_engine.save_trades_log()
        
        logger.info("✅ Trading system stopped")


_orchestrator_instance = None

def get_orchestrator(config: Dict = None) -> TradingOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None and config:
        _orchestrator_instance = TradingOrchestrator(config)
    return _orchestrator_instance
