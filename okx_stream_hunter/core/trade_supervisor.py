"""
Autonomous Trade Supervisor - PHASE 3
Monitors and validates all trading decisions, manages trade lifecycle
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeMonitor:
    """Monitor for active trade"""
    trade_id: str
    entry_time: datetime
    entry_price: float
    direction: str
    size: float
    sl: float
    tp: float
    confidence: float
    regime: str
    current_pnl: float = 0.0
    peak_pnl: float = 0.0
    trailing_sl: Optional[float] = None
    last_update: datetime = field(default_factory=datetime.now)


class TradeSupervisor:
    """Autonomous trade supervisor with self-healing capabilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.active_trades: Dict[str, TradeMonitor] = {}
        self.trade_history = deque(maxlen=1000)
        
        self.decision_buffer = deque(maxlen=100)
        self.pnl_history = deque(maxlen=500)
        
        self.stats = {
            'total_supervised': 0,
            'interventions': 0,
            'early_exits': 0,
            'trailing_stops': 0,
            'reversed_decisions': 0
        }
        
        self.min_time_between_trades = config.get('min_trade_interval_seconds', 30)
        self.last_trade_time = None
        
        self.trend_reversal_sensitivity = config.get('reversal_sensitivity', 0.8)
        self.trailing_sl_activation = config.get('trailing_sl_activation', 0.015)
        self.trailing_sl_distance = config.get('trailing_sl_distance', 0.01)
        
        logger.info("🛡️ Autonomous Trade Supervisor initialized")
    
    async def validate_decision(self, decision: Dict, market_state: Dict) -> Dict:
        """Validate AI decision before execution"""
        try:
            signal = decision.get('direction', 'NEUTRAL')
            confidence = decision.get('confidence', 0.0)
            regime = decision.get('regime', 'unknown')
            
            validation = {
                'approved': False,
                'reason': [],
                'modifications': {}
            }
            
            if signal == 'NEUTRAL':
                validation['reason'].append('No actionable signal')
                return validation
            
            if not self._check_time_interval():
                validation['reason'].append('Min time between trades not met')
                return validation
            
            if self._has_open_position_same_direction(signal):
                validation['reason'].append('Already have open position in same direction')
                return validation
            
            if self._detect_conflicting_signals():
                validation['reason'].append('Conflicting signals detected')
                validation['modifications']['reduce_size'] = 0.5
            
            if confidence < self.config.get('min_confidence_to_trade', 0.6):
                validation['reason'].append(f'Confidence too low: {confidence:.2%}')
                return validation
            
            if regime in self.config.get('blocked_regimes', []):
                validation['reason'].append(f'Regime blocked: {regime}')
                return validation
            
            trend_reversal_risk = await self._detect_trend_reversal(market_state)
            if trend_reversal_risk > self.trend_reversal_sensitivity:
                validation['reason'].append(f'Trend reversal risk: {trend_reversal_risk:.2%}')
                validation['modifications']['tighten_sl'] = 0.7
            
            validation['approved'] = True
            validation['reason'].append('All checks passed')
            
            logger.info(f"✅ Decision validated: {signal} @ {confidence:.2%}")
            return validation
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return {'approved': False, 'reason': [str(e)], 'modifications': {}}
    
    async def supervise_trade(self, trade_id: str, market_state: Dict) -> Optional[Dict]:
        """Monitor active trade and make autonomous decisions"""
        if trade_id not in self.active_trades:
            return None
        
        trade = self.active_trades[trade_id]
        current_price = market_state.get('price', 0.0)
        
        if current_price == 0:
            return None
        
        if trade.direction == 'LONG':
            trade.current_pnl = (current_price - trade.entry_price) / trade.entry_price
        else:
            trade.current_pnl = (trade.entry_price - current_price) / trade.entry_price
        
        trade.peak_pnl = max(trade.peak_pnl, trade.current_pnl)
        trade.last_update = datetime.now()
        
        self.pnl_history.append({
            'trade_id': trade_id,
            'timestamp': datetime.now(),
            'pnl': trade.current_pnl
        })
        
        action = None
        
        if self._check_sl_hit(trade, current_price):
            action = {'type': 'close', 'reason': 'SL hit', 'trade_id': trade_id}
            logger.warning(f"🛑 SL hit for trade {trade_id}")
        
        elif self._check_tp_hit(trade, current_price):
            action = {'type': 'close', 'reason': 'TP hit', 'trade_id': trade_id}
            logger.info(f"🎯 TP hit for trade {trade_id}")
        
        elif await self._should_activate_trailing_stop(trade, current_price):
            trade.trailing_sl = self._calculate_trailing_sl(trade, current_price)
            self.stats['trailing_stops'] += 1
            logger.info(f"📈 Trailing SL activated for {trade_id}: {trade.trailing_sl}")
        
        elif trade.trailing_sl and self._check_trailing_sl_hit(trade, current_price):
            action = {'type': 'close', 'reason': 'Trailing SL hit', 'trade_id': trade_id}
            logger.info(f"💰 Trailing SL hit for {trade_id}, locking profit")
        
        elif await self._detect_strong_reversal(trade, market_state):
            action = {'type': 'close', 'reason': 'Strong reversal detected', 'trade_id': trade_id}
            self.stats['early_exits'] += 1
            logger.warning(f"⚠️ Early exit due to reversal: {trade_id}")
        
        elif self._check_time_based_exit(trade):
            if trade.current_pnl < 0:
                action = {'type': 'close', 'reason': 'Time-based exit (loss)', 'trade_id': trade_id}
                logger.info(f"⏰ Time-based exit for {trade_id}")
        
        return action
    
    def register_trade(self, trade_data: Dict):
        """Register new trade for supervision"""
        trade_id = trade_data.get('trade_id', f"trade_{datetime.now().timestamp()}")
        
        monitor = TradeMonitor(
            trade_id=trade_id,
            entry_time=datetime.now(),
            entry_price=trade_data.get('entry_price', 0.0),
            direction=trade_data.get('direction', 'LONG'),
            size=trade_data.get('size', 0.0),
            sl=trade_data.get('sl', 0.0),
            tp=trade_data.get('tp', 0.0),
            confidence=trade_data.get('confidence', 0.0),
            regime=trade_data.get('regime', 'unknown')
        )
        
        self.active_trades[trade_id] = monitor
        self.stats['total_supervised'] += 1
        self.last_trade_time = datetime.now()
        
        logger.info(f"📝 Trade registered: {trade_id} | {monitor.direction} @ {monitor.entry_price}")
    
    def unregister_trade(self, trade_id: str, outcome: Dict):
        """Remove trade from active supervision"""
        if trade_id in self.active_trades:
            trade = self.active_trades.pop(trade_id)
            
            self.trade_history.append({
                'trade_id': trade_id,
                'entry_time': trade.entry_time,
                'exit_time': datetime.now(),
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'exit_price': outcome.get('exit_price', 0.0),
                'pnl': outcome.get('pnl', 0.0),
                'pnl_pct': outcome.get('pnl_pct', 0.0),
                'reason': outcome.get('reason', 'manual'),
                'peak_pnl': trade.peak_pnl,
                'confidence': trade.confidence,
                'regime': trade.regime
            })
            
            logger.info(f"✅ Trade unregistered: {trade_id} | PnL: {outcome.get('pnl_pct', 0):.2%}")
    
    def _check_time_interval(self) -> bool:
        """Check minimum time between trades"""
        if self.last_trade_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_trade_time).total_seconds()
        return elapsed >= self.min_time_between_trades
    
    def _has_open_position_same_direction(self, direction: str) -> bool:
        """Check if already have open position in same direction"""
        for trade in self.active_trades.values():
            if trade.direction == direction:
                return True
        return False
    
    def _detect_conflicting_signals(self) -> bool:
        """Detect if recent decisions are conflicting"""
        if len(self.decision_buffer) < 3:
            return False
        
        recent = list(self.decision_buffer)[-3:]
        signals = [d.get('direction') for d in recent]
        
        return len(set(signals)) == 3
    
    async def _detect_trend_reversal(self, market_state: Dict) -> float:
        """Detect potential trend reversal risk (0-1)"""
        try:
            buy_pressure = market_state.get('buy_pressure', 0.5)
            sell_pressure = market_state.get('sell_pressure', 0.5)
            imbalance = market_state.get('orderbook_imbalance', 0.5)
            
            pressure_shift = abs(buy_pressure - sell_pressure)
            imbalance_extreme = abs(imbalance - 0.5) * 2
            
            reversal_risk = (pressure_shift + imbalance_extreme) / 2
            
            if len(self.decision_buffer) >= 5:
                recent_directions = [d.get('direction') for d in list(self.decision_buffer)[-5:]]
                if len(set(recent_directions)) >= 3:
                    reversal_risk *= 1.5
            
            return min(reversal_risk, 1.0)
            
        except Exception:
            return 0.0
    
    def _check_sl_hit(self, trade: TradeMonitor, current_price: float) -> bool:
        """Check if stop loss hit"""
        if trade.direction == 'LONG':
            return current_price <= trade.sl
        else:
            return current_price >= trade.sl
    
    def _check_tp_hit(self, trade: TradeMonitor, current_price: float) -> bool:
        """Check if take profit hit"""
        if trade.direction == 'LONG':
            return current_price >= trade.tp
        else:
            return current_price <= trade.tp
    
    async def _should_activate_trailing_stop(self, trade: TradeMonitor, current_price: float) -> bool:
        """Check if should activate trailing stop"""
        if trade.trailing_sl is not None:
            return False
        
        return trade.current_pnl >= self.trailing_sl_activation
    
    def _calculate_trailing_sl(self, trade: TradeMonitor, current_price: float) -> float:
        """Calculate trailing stop loss level"""
        if trade.direction == 'LONG':
            return current_price * (1 - self.trailing_sl_distance)
        else:
            return current_price * (1 + self.trailing_sl_distance)
    
    def _check_trailing_sl_hit(self, trade: TradeMonitor, current_price: float) -> bool:
        """Check if trailing stop hit"""
        if trade.trailing_sl is None:
            return False
        
        if trade.direction == 'LONG':
            return current_price <= trade.trailing_sl
        else:
            return current_price >= trade.trailing_sl
    
    async def _detect_strong_reversal(self, trade: TradeMonitor, market_state: Dict) -> bool:
        """Detect strong trend reversal that justifies early exit"""
        buy_pressure = market_state.get('buy_pressure', 0.5)
        sell_pressure = market_state.get('sell_pressure', 0.5)
        
        if trade.direction == 'LONG':
            if sell_pressure > 0.75 and trade.current_pnl > 0:
                return True
        else:
            if buy_pressure > 0.75 and trade.current_pnl > 0:
                return True
        
        return False
    
    def _check_time_based_exit(self, trade: TradeMonitor) -> bool:
        """Check if should exit based on time"""
        max_hold_time = self.config.get('max_hold_time_minutes', 60)
        elapsed = (datetime.now() - trade.entry_time).total_seconds() / 60
        
        return elapsed >= max_hold_time
    
    def get_active_trades_status(self) -> List[Dict]:
        """Get status of all active trades"""
        return [
            {
                'trade_id': trade.trade_id,
                'direction': trade.direction,
                'entry_price': trade.entry_price,
                'current_pnl': trade.current_pnl,
                'peak_pnl': trade.peak_pnl,
                'sl': trade.sl,
                'tp': trade.tp,
                'trailing_sl': trade.trailing_sl,
                'confidence': trade.confidence,
                'regime': trade.regime,
                'age_seconds': (datetime.now() - trade.entry_time).total_seconds()
            }
            for trade in self.active_trades.values()
        ]
    
    def get_stats(self) -> Dict:
        """Get supervisor statistics"""
        return {
            **self.stats,
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.trade_history)
        }
