"""
Reinforcement Learning Self-Adaptive Agent
Learns from trade outcomes and adjusts strategy parameters
"""

import logging
from typing import Dict, List
from collections import deque
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RLAgent:
    """Self-adaptive RL agent that learns from trade outcomes"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.trade_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
        }
        
        self.adaptive_parameters = {
            'trend_weight': 1.0,
            'orderflow_weight': 1.0,
            'microstructure_weight': 1.0,
            'min_confidence_threshold': 0.6,
            'regime_multipliers': {
                'trend_up': 1.0,
                'trend_down': 1.0,
                'range': 1.0,
                'volatility_expansion': 1.0,
            }
        }
        
        self.pattern_performance = {}
        
        logger.info("🤖 RL Agent initialized")
    
    def update_from_trade(self, trade_result: Dict):
        """Learn from completed trade"""
        try:
            self.trade_history.append(trade_result)
            
            pnl = trade_result.get('pnl', 0.0)
            pattern = trade_result.get('pattern', 'unknown')
            regime = trade_result.get('regime', 'unknown')
            confidence = trade_result.get('confidence', 0.5)
            
            self.performance_metrics['total_trades'] += 1
            self.performance_metrics['total_pnl'] += pnl
            
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
            else:
                self.performance_metrics['losing_trades'] += 1
            
            self._update_pattern_stats(pattern, pnl, confidence)
            self._update_regime_stats(regime, pnl)
            self._recalculate_metrics()
            self._adapt_parameters()
            
            logger.info(f"📊 RL Update: PnL={pnl:.2f}, Pattern={pattern}, WinRate={self.performance_metrics['win_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"❌ RL update error: {e}")
    
    def _update_pattern_stats(self, pattern: str, pnl: float, confidence: float):
        """Track performance by pattern type"""
        if pattern not in self.pattern_performance:
            self.pattern_performance[pattern] = {
                'count': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0.0,
                'avg_confidence': 0.0,
                'weight': 1.0
            }
        
        stats = self.pattern_performance[pattern]
        stats['count'] += 1
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['wins'] += 1
        else:
            stats['losses'] += 1
        
        stats['avg_confidence'] = (stats['avg_confidence'] * (stats['count'] - 1) + confidence) / stats['count']
    
    def _update_regime_stats(self, regime: str, pnl: float):
        """Track performance by market regime"""
        if regime in self.adaptive_parameters['regime_multipliers']:
            current_multiplier = self.adaptive_parameters['regime_multipliers'][regime]
            
            if pnl > 0:
                self.adaptive_parameters['regime_multipliers'][regime] = min(1.5, current_multiplier + 0.05)
            elif pnl < 0:
                self.adaptive_parameters['regime_multipliers'][regime] = max(0.5, current_multiplier - 0.05)
    
    def _recalculate_metrics(self):
        """Recalculate performance metrics"""
        total = self.performance_metrics['total_trades']
        if total == 0:
            return
        
        wins = self.performance_metrics['winning_trades']
        losses = self.performance_metrics['losing_trades']
        
        self.performance_metrics['win_rate'] = wins / total if total > 0 else 0
        
        winning_trades = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        if winning_trades:
            self.performance_metrics['avg_win'] = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        
        if losing_trades:
            self.performance_metrics['avg_loss'] = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        
        total_wins = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_losses = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 1
        
        self.performance_metrics['profit_factor'] = total_wins / total_losses if total_losses > 0 else 0
    
    def _adapt_parameters(self):
        """Adapt strategy parameters based on recent performance"""
        recent_trades = list(self.trade_history)[-20:]
        
        if len(recent_trades) < 10:
            return
        
        recent_win_rate = sum(1 for t in recent_trades if t.get('pnl', 0) > 0) / len(recent_trades)
        
        if recent_win_rate > 0.65:
            self.adaptive_parameters['min_confidence_threshold'] = max(0.5, 
                self.adaptive_parameters['min_confidence_threshold'] - 0.02)
            logger.info(f"📈 Lowering confidence threshold to {self.adaptive_parameters['min_confidence_threshold']:.2f}")
        
        elif recent_win_rate < 0.40:
            self.adaptive_parameters['min_confidence_threshold'] = min(0.75, 
                self.adaptive_parameters['min_confidence_threshold'] + 0.02)
            logger.info(f"📉 Raising confidence threshold to {self.adaptive_parameters['min_confidence_threshold']:.2f}")
        
        for pattern, stats in self.pattern_performance.items():
            if stats['count'] >= 5:
                pattern_win_rate = stats['wins'] / stats['count']
                
                if pattern_win_rate > 0.6:
                    stats['weight'] = min(1.5, stats['weight'] + 0.05)
                elif pattern_win_rate < 0.4:
                    stats['weight'] = max(0.5, stats['weight'] - 0.05)
    
    def get_adaptive_confidence_threshold(self, regime: str, pattern: str) -> float:
        """Get adjusted confidence threshold for current context"""
        base_threshold = self.adaptive_parameters['min_confidence_threshold']
        
        regime_multiplier = self.adaptive_parameters['regime_multipliers'].get(regime, 1.0)
        
        pattern_weight = 1.0
        if pattern in self.pattern_performance:
            pattern_weight = self.pattern_performance[pattern]['weight']
        
        adjusted_threshold = base_threshold / (regime_multiplier * pattern_weight)
        
        return max(0.4, min(0.8, adjusted_threshold))
    
    def should_trade(self, decision: Dict, regime: str) -> bool:
        """Determine if trade should be executed based on learned parameters"""
        confidence = decision.get('confidence', 0.0)
        pattern = decision.get('pattern_type', 'unknown')
        
        threshold = self.get_adaptive_confidence_threshold(regime, pattern)
        
        if confidence < threshold:
            logger.info(f"❌ Confidence {confidence:.2f} below threshold {threshold:.2f}")
            return False
        
        if pattern in self.pattern_performance:
            stats = self.pattern_performance[pattern]
            if stats['count'] >= 10 and stats['wins'] / stats['count'] < 0.3:
                logger.info(f"❌ Pattern {pattern} has poor history")
                return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Get current RL stats"""
        return {
            'performance': self.performance_metrics.copy(),
            'adaptive_parameters': self.adaptive_parameters.copy(),
            'pattern_performance': {k: v.copy() for k, v in self.pattern_performance.items()},
            'recent_trades_count': len(self.trade_history)
        }
    
    def save_state(self, filepath: str = "rl_agent_state.json"):
        """Save RL agent state"""
        try:
            state = {
                'performance_metrics': self.performance_metrics,
                'adaptive_parameters': self.adaptive_parameters,
                'pattern_performance': self.pattern_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).write_text(json.dumps(state, indent=2))
            logger.info(f"💾 RL state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save RL state: {e}")
    
    def load_state(self, filepath: str = "rl_agent_state.json"):
        """Load RL agent state"""
        try:
            if not Path(filepath).exists():
                logger.info("No saved RL state found")
                return
            
            state = json.loads(Path(filepath).read_text())
            
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            self.adaptive_parameters = state.get('adaptive_parameters', self.adaptive_parameters)
            self.pattern_performance = state.get('pattern_performance', self.pattern_performance)
            
            logger.info(f"✅ RL state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load RL state: {e}")
