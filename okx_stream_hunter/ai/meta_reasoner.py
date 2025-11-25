"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Meta-Reasoning Omega Layer

Fuses all AI signals into final decision:
- CNN micro-patterns
- LSTM/Transformer sequences
- Orderflow intelligence
- RL agent recommendations
- Risk intelligence filters
- Regime-adaptive weighting
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FusionDecision:
    """Final AI decision output"""
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL', 'CLOSE'
    confidence: float
    regime: str
    sl: Optional[float]
    tp: Optional[float]
    reason: str
    timestamp: datetime
    features: Dict[str, float]
    component_votes: Dict[str, str]
    component_confidences: Dict[str, float]


class MetaReasoner:
    """Omega Layer - Supreme fusion intelligence"""
    
    def __init__(self, config):
        self.config = config
        self.decision_history = []
        logger.info("Initialized Meta-Reasoning Omega Layer")
    
    def fuse_signals(
        self,
        cnn_output,
        lstm_output,
        orderflow_output,
        rl_recommendation,
        regime: str,
        risk_filters: Dict,
        market_data: Dict
    ) -> FusionDecision:
        """
        Supreme fusion logic - combines all intelligence layers
        """
        
        # Collect component votes
        votes = {
            'cnn': self._map_direction(cnn_output.direction if cnn_output else 'neutral'),
            'lstm': self._map_direction(lstm_output.direction if lstm_output else 'neutral'),
            'orderflow': orderflow_output.signal.upper() if orderflow_output else 'NEUTRAL',
            'rl': rl_recommendation.get('action', 'NEUTRAL').upper() if rl_recommendation else 'NEUTRAL'
        }
        
        confidences = {
            'cnn': cnn_output.confidence if cnn_output else 0.0,
            'lstm': lstm_output.confidence if lstm_output else 0.0,
            'orderflow': orderflow_output.confidence if orderflow_output else 0.0,
            'rl': rl_recommendation.get('confidence', 0.0) if rl_recommendation else 0.0
        }
        
        # Get regime-adaptive weights
        weights = self._get_adaptive_weights(regime)
        
        # Weighted ensemble fusion
        if self.config.fusion_method == "weighted_ensemble":
            direction, confidence = self._weighted_ensemble(votes, confidences, weights)
        else:
            # Neural fusion (future enhancement)
            direction, confidence = self._weighted_ensemble(votes, confidences, weights)
        
        # Apply risk filters
        direction, confidence = self._apply_risk_filters(
            direction, confidence, risk_filters, market_data
        )
        
        # Generate SL/TP
        sl, tp = self._calculate_sl_tp(direction, market_data, risk_filters)
        
        # Generate reasoning
        reason = self._generate_reasoning(votes, confidences, regime, risk_filters)
        
        # Aggregate features
        features = self._aggregate_features(cnn_output, lstm_output, orderflow_output)
        
        decision = FusionDecision(
            direction=direction,
            confidence=confidence,
            regime=regime,
            sl=sl,
            tp=tp,
            reason=reason,
            timestamp=datetime.now(),
            features=features,
            component_votes=votes,
            component_confidences=confidences
        )
        
        self.decision_history.append(decision)
        return decision
    
    def _map_direction(self, direction: str) -> str:
        """Map various direction formats to standard"""
        direction = str(direction).lower()
        if 'bull' in direction or 'long' in direction or 'buy' in direction or direction == '1':
            return 'LONG'
        elif 'bear' in direction or 'short' in direction or 'sell' in direction or direction == '-1':
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    def _get_adaptive_weights(self, regime: str) -> Dict[str, float]:
        """Regime-adaptive component weights"""
        base_weights = self.config.weights.copy()
        
        # Adjust weights based on regime
        if regime == 'trend_up' or regime == 'trend_down':
            # Trust LSTM more in trends
            base_weights['lstm'] *= 1.2
            base_weights['orderflow'] *= 0.9
        elif regime == 'range':
            # Trust orderflow more in ranging markets
            base_weights['orderflow'] *= 1.3
            base_weights['lstm'] *= 0.8
        elif regime == 'volatility_expansion':
            # Trust CNN patterns more in volatile markets
            base_weights['cnn'] *= 1.2
            base_weights['rl'] *= 1.1
        elif regime == 'liquidity_sweep' or regime == 'breakout':
            # Trust orderflow heavily during liquidity events
            base_weights['orderflow'] *= 1.5
            base_weights['rl'] *= 1.2
        
        # Normalize weights
        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()}
    
    def _weighted_ensemble(
        self,
        votes: Dict[str, str],
        confidences: Dict[str, float],
        weights: Dict[str, float]
    ) -> Tuple[str, float]:
        """Weighted ensemble voting"""
        
        # Calculate weighted scores
        long_score = sum(
            weights.get(comp, 0) * confidences.get(comp, 0)
            for comp, vote in votes.items()
            if vote == 'LONG'
        )
        
        short_score = sum(
            weights.get(comp, 0) * confidences.get(comp, 0)
            for comp, vote in votes.items()
            if vote == 'SHORT'
        )
        
        neutral_score = sum(
            weights.get(comp, 0) * confidences.get(comp, 0)
            for comp, vote in votes.items()
            if vote == 'NEUTRAL'
        )
        
        # Determine direction
        max_score = max(long_score, short_score, neutral_score)
        
        if max_score < self.config.min_confidence_threshold:
            return 'NEUTRAL', max_score
        
        if long_score == max_score:
            direction = 'LONG'
        elif short_score == max_score:
            direction = 'SHORT'
        else:
            direction = 'NEUTRAL'
        
        # Confidence is the max score
        confidence = min(max_score, 1.0)
        
        return direction, confidence
    
    def _apply_risk_filters(
        self,
        direction: str,
        confidence: float,
        risk_filters: Dict,
        market_data: Dict
    ) -> Tuple[str, float]:
        """Apply risk intelligence filters"""
        
        # Block if dangerous conditions
        if risk_filters.get('block_trading', False):
            logger.warning("Risk filter BLOCKING trade")
            return 'NEUTRAL', 0.0
        
        # Reduce confidence if high volatility
        if risk_filters.get('high_volatility', False):
            confidence *= 0.7
            logger.info("High volatility - reducing confidence")
        
        # Block if liquidation clusters nearby
        if risk_filters.get('liquidation_danger', False):
            logger.warning("Liquidation danger - neutralizing signal")
            return 'NEUTRAL', 0.0
        
        # Reduce confidence if orderbook imbalance is dangerous
        if risk_filters.get('dangerous_imbalance', False):
            confidence *= 0.5
            logger.info("Dangerous imbalance - reducing confidence")
        
        return direction, confidence
    
    def _calculate_sl_tp(
        self,
        direction: str,
        market_data: Dict,
        risk_filters: Dict
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop-loss and take-profit"""
        
        if direction == 'NEUTRAL':
            return None, None
        
        current_price = market_data.get('price', 0)
        atr = risk_filters.get('atr', current_price * 0.01)  # Fallback 1% ATR
        
        sl_distance = atr * self.config.sl_atr_multiplier
        tp_distance = sl_distance * self.config.tp_risk_reward_ratio
        
        if direction == 'LONG':
            sl = current_price - sl_distance
            tp = current_price + tp_distance
        else:  # SHORT
            sl = current_price + sl_distance
            tp = current_price - tp_distance
        
        return sl, tp
    
    def _generate_reasoning(
        self,
        votes: Dict[str, str],
        confidences: Dict[str, float],
        regime: str,
        risk_filters: Dict
    ) -> str:
        """Generate human-readable reasoning"""
        
        reasons = []
        
        # Component agreements
        long_voters = [k for k, v in votes.items() if v == 'LONG']
        short_voters = [k for k, v in votes.items() if v == 'SHORT']
        
        if len(long_voters) >= 3:
            reasons.append(f"Strong consensus: {', '.join(long_voters)} agree on LONG")
        elif len(short_voters) >= 3:
            reasons.append(f"Strong consensus: {', '.join(short_voters)} agree on SHORT")
        
        # Regime context
        reasons.append(f"Regime: {regime}")
        
        # Top confidence components
        sorted_confs = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        if sorted_confs:
            top = sorted_confs[0]
            reasons.append(f"{top[0].upper()} confidence: {top[1]:.2%}")
        
        # Risk considerations
        if risk_filters.get('high_volatility'):
            reasons.append("High volatility detected")
        if risk_filters.get('liquidation_danger'):
            reasons.append("Liquidation clusters nearby")
        
        return " | ".join(reasons)
    
    def _aggregate_features(self, cnn_output, lstm_output, orderflow_output) -> Dict[str, float]:
        """Aggregate features from all components"""
        features = {}
        
        if cnn_output:
            features['cnn_confidence'] = cnn_output.confidence
            if hasattr(cnn_output, 'pattern_type'):
                features['pattern'] = cnn_output.pattern_type
        
        if lstm_output:
            features['lstm_confidence'] = lstm_output.confidence
            if hasattr(lstm_output, 'predicted_move'):
                features['predicted_move'] = lstm_output.predicted_move
        
        if orderflow_output:
            features['orderflow_confidence'] = orderflow_output.confidence
            features.update(orderflow_output.metrics)
        
        return features
    
    def get_recent_decisions(self, count: int = 10) -> List[FusionDecision]:
        """Get recent decision history"""
        return self.decision_history[-count:]
