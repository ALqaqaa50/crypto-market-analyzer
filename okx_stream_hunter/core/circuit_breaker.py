"""
Circuit Breaker - PHASE 3
Emergency stop mechanism with daily loss limits and auto-recovery
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Trading circuit breaker with multiple trigger conditions"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.is_open = False
        self.open_reason = None
        self.open_timestamp = None
        
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.starting_balance = config.get('initial_balance', 10000.0)
        
        self.max_daily_loss_pct = config.get('max_daily_drawdown', 0.10)
        self.max_daily_loss_amount = self.starting_balance * self.max_daily_loss_pct
        
        self.max_single_loss_pct = config.get('max_single_loss_pct', 0.05)
        self.max_daily_trades = config.get('max_daily_trades', 20)
        
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.consecutive_losses = 0
        
        self.trade_history_today = deque(maxlen=100)
        self.daily_reset_hour = config.get('daily_reset_hour', 0)
        
        self.auto_reset_enabled = config.get('circuit_auto_reset', False)
        self.auto_reset_delay_minutes = config.get('circuit_reset_delay', 60)
        
        self.trip_count = 0
        self.last_reset = datetime.now()
        
        logger.info(f"⚡ Circuit Breaker initialized: {self.max_daily_loss_pct:.1%} daily limit")
    
    async def check(self, before_trade: bool = True) -> Dict:
        """Check if circuit should trip"""
        result = {
            'allowed': True,
            'reason': None,
            'warnings': []
        }
        
        if self.is_open:
            result['allowed'] = False
            result['reason'] = f'Circuit breaker is open: {self.open_reason}'
            return result
        
        await self._check_daily_reset()
        
        if before_trade:
            if self.daily_trades >= self.max_daily_trades:
                await self._trip('max_daily_trades_reached')
                result['allowed'] = False
                result['reason'] = f'Max daily trades reached: {self.daily_trades}/{self.max_daily_trades}'
                return result
        
        if abs(self.daily_pnl) >= self.max_daily_loss_amount:
            await self._trip('max_daily_loss_exceeded')
            result['allowed'] = False
            result['reason'] = f'Max daily loss exceeded: ${abs(self.daily_pnl):.2f}'
            return result
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            await self._trip('max_consecutive_losses')
            result['allowed'] = False
            result['reason'] = f'Max consecutive losses: {self.consecutive_losses}'
            return result
        
        loss_remaining = self.max_daily_loss_amount - abs(self.daily_pnl)
        if loss_remaining < self.max_daily_loss_amount * 0.2:
            result['warnings'].append(f'Approaching daily loss limit: ${loss_remaining:.2f} remaining')
        
        trades_remaining = self.max_daily_trades - self.daily_trades
        if trades_remaining <= 3:
            result['warnings'].append(f'Approaching trade limit: {trades_remaining} trades remaining')
        
        if self.consecutive_losses >= 3:
            result['warnings'].append(f'Consecutive losses: {self.consecutive_losses}')
        
        return result
    
    async def record_trade(self, trade_result: Dict):
        """Record trade result and update circuit state"""
        pnl = trade_result.get('pnl', 0.0)
        pnl_pct = trade_result.get('pnl_pct', 0.0)
        
        self.daily_pnl += pnl
        self.daily_trades += 1
        
        self.trade_history_today.append({
            'timestamp': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'daily_pnl': self.daily_pnl
        })
        
        if pnl < 0:
            self.consecutive_losses += 1
            
            if abs(pnl_pct) >= self.max_single_loss_pct:
                await self._trip('single_loss_too_large')
                logger.critical(f"🚨 Circuit tripped: Single loss {pnl_pct:.2%} exceeds {self.max_single_loss_pct:.2%}")
        else:
            self.consecutive_losses = 0
        
        logger.info(
            f"📊 Circuit Update: Daily PnL=${self.daily_pnl:.2f} | "
            f"Trades={self.daily_trades}/{self.max_daily_trades} | "
            f"Consecutive Losses={self.consecutive_losses}"
        )
    
    async def _trip(self, reason: str):
        """Trip the circuit breaker"""
        if self.is_open:
            return
        
        self.is_open = True
        self.open_reason = reason
        self.open_timestamp = datetime.now()
        self.trip_count += 1
        
        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        logger.critical(f"📊 Daily Stats: PnL=${self.daily_pnl:.2f}, Trades={self.daily_trades}")
        
        if self.auto_reset_enabled:
            asyncio.create_task(self._auto_reset())
    
    async def _auto_reset(self):
        """Auto-reset circuit after delay"""
        delay_seconds = self.auto_reset_delay_minutes * 60
        logger.info(f"⏱️ Circuit will auto-reset in {self.auto_reset_delay_minutes} minutes")
        
        await asyncio.sleep(delay_seconds)
        
        if self.is_open:
            await self.reset()
    
    async def reset(self, force: bool = False):
        """Reset circuit breaker"""
        if not self.is_open and not force:
            logger.warning("⚠️ Circuit already closed")
            return
        
        old_reason = self.open_reason
        
        self.is_open = False
        self.open_reason = None
        self.open_timestamp = None
        self.last_reset = datetime.now()
        
        logger.info(f"✅ Circuit breaker reset (was: {old_reason})")
    
    async def _check_daily_reset(self):
        """Check if should reset daily counters"""
        now = datetime.now()
        
        if now.hour == self.daily_reset_hour and now.date() > self.last_reset.date():
            await self._reset_daily_counters()
    
    async def _reset_daily_counters(self):
        """Reset daily trading counters"""
        logger.info("🔄 Daily reset: counters cleared")
        
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.trade_history_today.clear()
        self.last_reset = datetime.now()
        
        if self.is_open and self.open_reason in ['max_daily_trades_reached', 'max_daily_loss_exceeded']:
            await self.reset()
    
    def force_daily_reset(self):
        """Manually force daily reset"""
        logger.info("🔄 Forced daily reset")
        asyncio.create_task(self._reset_daily_counters())
    
    def get_status(self) -> Dict:
        """Get circuit breaker status"""
        loss_remaining = self.max_daily_loss_amount - abs(self.daily_pnl)
        loss_pct_used = (abs(self.daily_pnl) / self.max_daily_loss_amount) * 100 if self.max_daily_loss_amount > 0 else 0
        
        trades_remaining = self.max_daily_trades - self.daily_trades
        trades_pct_used = (self.daily_trades / self.max_daily_trades) * 100 if self.max_daily_trades > 0 else 0
        
        status = {
            'is_open': self.is_open,
            'open_reason': self.open_reason,
            'open_since': self.open_timestamp.isoformat() if self.open_timestamp else None,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'max_daily_trades': self.max_daily_trades,
            'trades_remaining': trades_remaining,
            'trades_pct_used': trades_pct_used,
            'max_daily_loss': self.max_daily_loss_amount,
            'loss_remaining': loss_remaining,
            'loss_pct_used': loss_pct_used,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'trip_count': self.trip_count,
            'last_reset': self.last_reset.isoformat()
        }
        
        if loss_pct_used >= 80 or trades_pct_used >= 80:
            status['risk_level'] = 'high'
        elif loss_pct_used >= 50 or trades_pct_used >= 50:
            status['risk_level'] = 'medium'
        else:
            status['risk_level'] = 'low'
        
        return status
    
    def get_health_score(self) -> float:
        """Get circuit health score (0-1)"""
        if self.is_open:
            return 0.0
        
        score = 1.0
        
        loss_pct = abs(self.daily_pnl) / self.max_daily_loss_amount if self.max_daily_loss_amount > 0 else 0
        score -= loss_pct * 0.4
        
        trades_pct = self.daily_trades / self.max_daily_trades if self.max_daily_trades > 0 else 0
        score -= trades_pct * 0.3
        
        loss_streak_pct = self.consecutive_losses / self.max_consecutive_losses if self.max_consecutive_losses > 0 else 0
        score -= loss_streak_pct * 0.3
        
        return max(0.0, min(1.0, score))
