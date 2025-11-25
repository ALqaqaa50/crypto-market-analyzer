"""
PHASE 4: Experience Buffer
Thread-safe in-memory buffer for training experiences
"""

import threading
from collections import deque
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Experience:
    """Single experience/step data structure"""
    
    def __init__(
        self,
        timestamp: datetime,
        symbol: str,
        market_features: Dict[str, Any],
        ai_decision: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None,
        trade_outcome: Optional[Dict[str, Any]] = None,
        risk_context: Optional[Dict[str, Any]] = None
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.market_features = market_features
        self.ai_decision = ai_decision
        self.execution_result = execution_result
        self.trade_outcome = trade_outcome
        self.risk_context = risk_context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'market_features': self.market_features,
            'ai_decision': self.ai_decision,
            'execution_result': self.execution_result,
            'trade_outcome': self.trade_outcome,
            'risk_context': self.risk_context
        }


class ExperienceBuffer:
    """
    Thread-safe circular buffer for storing experiences
    Used for both immediate logging and RL training
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        self.stats = {
            'total_added': 0,
            'total_decisions': 0,
            'total_trades': 0,
            'buffer_size': 0
        }
        
        logger.info(f"📦 ExperienceBuffer initialized (max_size={max_size})")
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        with self.lock:
            self.buffer.append(experience)
            self.stats['total_added'] += 1
            self.stats['buffer_size'] = len(self.buffer)
            
            if experience.ai_decision:
                self.stats['total_decisions'] += 1
            
            if experience.execution_result:
                self.stats['total_trades'] += 1
    
    def add_decision(
        self,
        symbol: str,
        market_features: Dict[str, Any],
        ai_decision: Dict[str, Any],
        risk_context: Optional[Dict[str, Any]] = None
    ):
        """Convenience method to add decision-only experience"""
        experience = Experience(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            market_features=market_features,
            ai_decision=ai_decision,
            risk_context=risk_context
        )
        self.add(experience)
    
    def add_trade_outcome(
        self,
        symbol: str,
        market_features: Dict[str, Any],
        ai_decision: Dict[str, Any],
        execution_result: Dict[str, Any],
        trade_outcome: Dict[str, Any],
        risk_context: Optional[Dict[str, Any]] = None
    ):
        """Add complete trade experience with outcome"""
        experience = Experience(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            market_features=market_features,
            ai_decision=ai_decision,
            execution_result=execution_result,
            trade_outcome=trade_outcome,
            risk_context=risk_context
        )
        self.add(experience)
    
    def get_recent(self, n: int = 100) -> List[Experience]:
        """Get N most recent experiences"""
        with self.lock:
            if n >= len(self.buffer):
                return list(self.buffer)
            return list(self.buffer)[-n:]
    
    def get_all(self) -> List[Experience]:
        """Get all experiences in buffer"""
        with self.lock:
            return list(self.buffer)
    
    def get_trades_only(self) -> List[Experience]:
        """Get only experiences with actual trade executions"""
        with self.lock:
            return [exp for exp in self.buffer if exp.execution_result is not None]
    
    def clear(self):
        """Clear all experiences (use with caution)"""
        with self.lock:
            self.buffer.clear()
            self.stats['buffer_size'] = 0
            logger.warning("⚠️ ExperienceBuffer cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return self.stats.copy()
    
    def export_to_list(self) -> List[Dict[str, Any]]:
        """Export all experiences as list of dicts"""
        with self.lock:
            return [exp.to_dict() for exp in self.buffer]


# Global instance
_experience_buffer: Optional[ExperienceBuffer] = None


def get_experience_buffer() -> ExperienceBuffer:
    """Get global experience buffer instance"""
    global _experience_buffer
    if _experience_buffer is None:
        _experience_buffer = ExperienceBuffer(max_size=10000)
    return _experience_buffer


def reset_experience_buffer(max_size: int = 10000):
    """Reset global buffer (for testing)"""
    global _experience_buffer
    _experience_buffer = ExperienceBuffer(max_size=max_size)
