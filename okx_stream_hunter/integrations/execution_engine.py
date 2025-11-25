"""
Execution Engine - Handle order execution (Paper Trading + Live Trading)
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from okx_stream_hunter.storage.trade_logger import get_trade_logger
from okx_stream_hunter.core.experience_buffer import get_experience_buffer
from okx_stream_hunter.core.trading_mode import get_trading_mode_manager
from okx_stream_hunter.notifications import get_telegram_client

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class ExecutionEngine:
    """Order execution engine supporting paper and live trading"""
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.PAPER, config: Dict = None):
        self.mode = mode
        self.config = config or {}
        
        self.open_positions = {}
        self.closed_trades = []
        self.paper_balance = self.config.get('initial_balance', 10000.0)
        self.paper_equity = self.paper_balance
        
        self.execution_log = []
        
        self.trade_logger = get_trade_logger()
        self.experience_buffer = get_experience_buffer()
        
        self.trading_mode_manager = get_trading_mode_manager(config.get('trading_mode'))
        self.telegram = get_telegram_client()
        
        mode_prefix = self.trading_mode_manager.get_log_prefix()
        logger.info(f"{mode_prefix} ⚡ Execution Engine initialized in {mode.value.upper()} mode")
        logger.info(f"{mode_prefix} 💰 Initial balance: ${self.paper_balance:,.2f}")
    
    async def execute_signal(
        self,
        decision: Dict,
        position_size: float,
        market_state: Dict
    ) -> Optional[Dict]:
        """Execute trading signal"""
        try:
            signal = decision.get('direction', 'NEUTRAL')
            
            if signal == 'NEUTRAL':
                return None
            
            current_price = market_state.get('price', 0.0)
            sl_price = decision.get('sl', 0.0)
            tp_price = decision.get('tp', 0.0)
            confidence = decision.get('confidence', 0.0)
            
            if self.mode == ExecutionMode.PAPER:
                result = await self._execute_paper_trade(
                    signal, position_size, current_price, sl_price, tp_price, confidence, decision
                )
            else:
                result = await self._execute_live_trade(
                    signal, position_size, current_price, sl_price, tp_price, confidence, decision
                )
            
            if result:
                self.execution_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'signal': signal,
                    'price': current_price,
                    'size': position_size,
                    'mode': self.mode.value,
                    'result': result
                })
                
                try:
                    trading_mode = self.trading_mode_manager.mode.value
                    self.telegram.send_message_sync(
                        f"{self.trading_mode_manager.get_log_prefix()} Trade executed: "
                        f"{signal} {position_size:.4f} @ ${current_price:,.2f}"
                    )
                except Exception as e:
                    logger.error(f"Telegram notification failed: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return None
    
    async def _execute_paper_trade(
        self,
        signal: str,
        size: float,
        price: float,
        sl: float,
        tp: float,
        confidence: float,
        decision: Dict
    ) -> Dict:
        """Execute paper trade"""
        try:
            if signal == 'CLOSE' and self.open_positions:
                return await self._close_paper_positions(price)
            
            if self.open_positions:
                logger.info("⚠️ Position already open, skipping new entry")
                return None
            
            position_id = f"PAPER_{datetime.now().timestamp()}"
            
            position = {
                'id': position_id,
                'signal': signal,
                'entry_price': price,
                'size': size,
                'sl': sl,
                'tp': tp,
                'confidence': confidence,
                'entry_time': datetime.now(),
                'status': 'open',
                'regime': decision.get('regime', 'unknown'),
                'pattern': decision.get('pattern_type', 'unknown'),
                'reason': decision.get('reason', '')
            }
            
            required_margin = size * price
            
            if required_margin > self.paper_balance * 0.95:
                logger.warning(f"❌ Insufficient balance: {self.paper_balance:.2f} < {required_margin:.2f}")
                return None
            
            self.open_positions[position_id] = position
            self.paper_balance -= required_margin * 0.01
            
            logger.info(f"✅ PAPER TRADE OPENED: {signal} {size} @ {price:.2f}")
            logger.info(f"   SL: {sl:.2f} | TP: {tp:.2f} | Confidence: {confidence:.2%}")
            logger.info(f"   Balance: ${self.paper_balance:,.2f}")
            
            # PHASE 4: Log trade execution
            self._log_trade_execution(position, decision, {'price': price})
            
            return position
            
        except Exception as e:
            logger.error(f"❌ Paper trade error: {e}")
            return None
    
    async def _close_paper_positions(self, current_price: float) -> Dict:
        """Close all paper positions"""
        results = []
        
        for pos_id, position in list(self.open_positions.items()):
            entry_price = position['entry_price']
            size = position['size']
            signal = position['signal']
            
            if signal == 'LONG':
                pnl = (current_price - entry_price) * size
            else:
                pnl = (entry_price - current_price) * size
            
            pnl_pct = (pnl / (entry_price * size)) * 100
            
            position['exit_price'] = current_price
            position['exit_time'] = datetime.now()
            position['pnl'] = pnl
            position['pnl_pct'] = pnl_pct
            position['status'] = 'closed'
            
            self.paper_balance += (entry_price * size) + pnl
            self.paper_equity = self.paper_balance
            
            self.closed_trades.append(position)
            del self.open_positions[pos_id]
            
            logger.info(f"🔒 PAPER TRADE CLOSED: {signal} PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
            logger.info(f"   Balance: ${self.paper_balance:,.2f}")
            
            # PHASE 4: Log trade outcome
            self._log_trade_outcome(position, current_price, pnl, pnl_pct)
            
            results.append(position)
        
        return {'closed_positions': results}
    
    async def _execute_live_trade(
        self,
        signal: str,
        size: float,
        price: float,
        sl: float,
        tp: float,
        confidence: float,
        decision: Dict
    ) -> Dict:
        """Execute live trade (placeholder for OKX API integration)"""
        logger.warning("⚠️ Live trading not yet implemented - use PAPER mode")
        return None
    
    async def check_stop_loss_take_profit(self, current_price: float) -> Optional[Dict]:
        """Check if any positions hit SL/TP"""
        if not self.open_positions:
            return None
        
        for pos_id, position in list(self.open_positions.items()):
            signal = position['signal']
            sl = position['sl']
            tp = position['tp']
            
            hit_sl = False
            hit_tp = False
            
            if signal == 'LONG':
                if current_price <= sl:
                    hit_sl = True
                elif current_price >= tp:
                    hit_tp = True
            else:
                if current_price >= sl:
                    hit_sl = True
                elif current_price <= tp:
                    hit_tp = True
            
            if hit_sl:
                logger.info(f"🛑 Stop Loss HIT at {current_price:.2f}")
                return await self._close_paper_positions(current_price)
            
            if hit_tp:
                logger.info(f"🎯 Take Profit HIT at {current_price:.2f}")
                return await self._close_paper_positions(current_price)
        
        return None
    
    def get_open_positions(self) -> Dict:
        """Get all open positions"""
        return self.open_positions.copy()
    
    def get_closed_trades(self) -> list:
        """Get all closed trades"""
        return self.closed_trades.copy()
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.get('pnl', 0) > 0)
        
        total_pnl = sum(t.get('pnl', 0) for t in self.closed_trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'mode': self.mode.value,
            'balance': self.paper_balance,
            'equity': self.paper_equity,
            'open_positions': len(self.open_positions),
            'closed_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / 10000) * 100 if self.config.get('initial_balance', 10000) > 0 else 0
        }
    
    def _log_trade_execution(self, position: Dict, decision: Dict, market_state: Dict):
        """PHASE 4: Log trade execution to disk"""
        try:
            market_features = {
                'price': market_state.get('price', 0),
                'bid': market_state.get('bid', 0),
                'ask': market_state.get('ask', 0)
            }
            
            ai_decision = {
                'direction': position['signal'],
                'confidence': position['confidence'],
                'regime': position['regime'],
                'reason': position['reason']
            }
            
            execution_result = {
                'trade_id': position['id'],
                'order_id': position['id'],
                'filled_size': position['size'],
                'avg_fill_price': position['entry_price'],
                'sl': position['sl'],
                'tp': position['tp']
            }
            
            risk_context = {
                'equity': self.paper_equity,
                'daily_pnl': 0,  # Placeholder
                'open_positions': len(self.open_positions)
            }
            
            self.trade_logger.log_trade(
                timestamp=datetime.utcnow(),
                symbol='BTC-USDT-SWAP',
                market_features=market_features,
                ai_decision=ai_decision,
                execution_result=execution_result,
                risk_context=risk_context
            )
        
        except Exception as e:
            logger.error(f"Error logging trade execution: {e}")
    
    def _log_trade_outcome(self, position: Dict, exit_price: float, pnl: float, pnl_pct: float):
        """PHASE 4: Log trade outcome to disk"""
        try:
            duration = (position['exit_time'] - position['entry_time']).total_seconds()
            
            self.trade_logger.log_trade_outcome(
                timestamp=datetime.utcnow(),
                symbol='BTC-USDT-SWAP',
                trade_id=position['id'],
                entry_price=position['entry_price'],
                exit_price=exit_price,
                direction=position['signal'],
                size=position['size'],
                pnl=pnl,
                pnl_pct=pnl_pct,
                duration_seconds=duration,
                exit_reason='manual_close'  # Placeholder
            )
        
        except Exception as e:
            logger.error(f"Error logging trade outcome: {e}")
    
    def save_trades_log(self, filepath: str = "trades_log.json"):
        """Save trades log to file"""
        try:
            log_data = {
                'execution_log': self.execution_log,
                'closed_trades': self.closed_trades,
                'stats': self.get_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).write_text(json.dumps(log_data, indent=2, default=str))
            logger.info(f"💾 Trades log saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save trades log: {e}")
