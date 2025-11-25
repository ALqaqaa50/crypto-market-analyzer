import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class OmegaFusionEngine:
    def __init__(self, regime_weight_profiles: Dict = None):
        self.signal_weights = {
            'cnn': 0.15,
            'lstm': 0.15,
            'orderflow': 0.20,
            'rl': 0.25,
            'meta_reasoner': 0.15,
            'risk': 0.10
        }
        
        self.regime_weight_profiles = regime_weight_profiles or self._default_regime_profiles()
        
        self.signal_history = {key: deque(maxlen=1000) for key in self.signal_weights.keys()}
        self.signal_performance = {key: {'correct': 0, 'total': 0, 'avg_confidence': 0.0} 
                                   for key in self.signal_weights.keys()}
        
        self.fusion_history = deque(maxlen=5000)
        self.last_regime = 'unknown'
        
        logger.info("OmegaFusionEngine initialized with adaptive signal weighting")
    
    def fuse_signals(self, signals: Dict, regime: str = None, market_state: Dict = None) -> Dict:
        if regime and regime in self.regime_weight_profiles:
            self.signal_weights = self.regime_weight_profiles[regime].copy()
            self.last_regime = regime
        
        normalized_signals = self._normalize_signals(signals)
        
        confidences = self._extract_confidences(signals)
        
        weighted_signals = {}
        total_weight = 0.0
        
        for signal_name, signal_value in normalized_signals.items():
            if signal_name in self.signal_weights:
                base_weight = self.signal_weights[signal_name]
                confidence = confidences.get(signal_name, 1.0)
                
                performance = self.signal_performance[signal_name]
                accuracy = performance['correct'] / max(performance['total'], 1)
                
                adjusted_weight = base_weight * confidence * (0.5 + 0.5 * accuracy)
                
                weighted_signals[signal_name] = {
                    'value': signal_value,
                    'weight': adjusted_weight,
                    'confidence': confidence
                }
                
                total_weight += adjusted_weight
        
        if total_weight == 0:
            return {
                'action': 0.0,
                'confidence': 0.0,
                'signal_contributions': {},
                'regime': regime or 'unknown'
            }
        
        fused_action = 0.0
        signal_contributions = {}
        
        for signal_name, signal_data in weighted_signals.items():
            normalized_weight = signal_data['weight'] / total_weight
            contribution = signal_data['value'] * normalized_weight
            
            fused_action += contribution
            signal_contributions[signal_name] = {
                'contribution': contribution,
                'weight': normalized_weight,
                'raw_value': signal_data['value'],
                'confidence': signal_data['confidence']
            }
        
        overall_confidence = self._calculate_overall_confidence(weighted_signals, total_weight)
        
        if market_state:
            fused_action = self._apply_risk_constraints(fused_action, market_state, overall_confidence)
        
        fusion_result = {
            'action': np.clip(fused_action, -1.0, 1.0),
            'confidence': overall_confidence,
            'signal_contributions': signal_contributions,
            'regime': regime or self.last_regime,
            'total_signals': len(weighted_signals)
        }
        
        self.fusion_history.append(fusion_result)
        
        for signal_name, signal_data in normalized_signals.items():
            if signal_name in self.signal_history:
                self.signal_history[signal_name].append(signal_data)
        
        return fusion_result
    
    def _normalize_signals(self, signals: Dict) -> Dict:
        normalized = {}
        
        for key, value in signals.items():
            if isinstance(value, dict):
                if 'action' in value:
                    normalized[key] = np.clip(value['action'], -1.0, 1.0)
                elif 'signal' in value:
                    normalized[key] = np.clip(value['signal'], -1.0, 1.0)
                elif 'prediction' in value:
                    normalized[key] = np.clip(value['prediction'], -1.0, 1.0)
                else:
                    normalized[key] = 0.0
            elif isinstance(value, (int, float)):
                normalized[key] = np.clip(value, -1.0, 1.0)
            else:
                normalized[key] = 0.0
        
        return normalized
    
    def _extract_confidences(self, signals: Dict) -> Dict:
        confidences = {}
        
        for key, value in signals.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidences[key] = np.clip(value['confidence'], 0.0, 1.0)
            else:
                confidences[key] = 1.0
        
        return confidences
    
    def _calculate_overall_confidence(self, weighted_signals: Dict, total_weight: float) -> float:
        if not weighted_signals:
            return 0.0
        
        confidence_sum = 0.0
        for signal_data in weighted_signals.values():
            confidence_sum += signal_data['confidence'] * signal_data['weight']
        
        overall_confidence = confidence_sum / total_weight if total_weight > 0 else 0.0
        
        agreement_score = self._calculate_signal_agreement(weighted_signals)
        
        final_confidence = overall_confidence * (0.7 + 0.3 * agreement_score)
        
        return np.clip(final_confidence, 0.0, 1.0)
    
    def _calculate_signal_agreement(self, weighted_signals: Dict) -> float:
        if len(weighted_signals) < 2:
            return 1.0
        
        values = [s['value'] for s in weighted_signals.values()]
        
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if std_value < 0.1:
            return 1.0
        elif std_value < 0.3:
            return 0.7
        elif std_value < 0.5:
            return 0.5
        else:
            return 0.3
    
    def _apply_risk_constraints(self, action: float, market_state: Dict, confidence: float) -> float:
        risk_multiplier = 1.0
        
        if 'volatility' in market_state:
            volatility = market_state['volatility']
            if volatility > 0.03:
                risk_multiplier *= 0.5
            elif volatility > 0.02:
                risk_multiplier *= 0.7
        
        if 'drawdown' in market_state:
            drawdown = abs(market_state['drawdown'])
            if drawdown > 0.15:
                risk_multiplier *= 0.3
            elif drawdown > 0.10:
                risk_multiplier *= 0.6
        
        if 'recent_losses' in market_state:
            if market_state['recent_losses'] >= 3:
                risk_multiplier *= 0.5
        
        if confidence < 0.5:
            risk_multiplier *= confidence
        
        return action * risk_multiplier
    
    def update_signal_performance(self, signal_name: str, was_correct: bool, confidence: float = 1.0):
        if signal_name in self.signal_performance:
            perf = self.signal_performance[signal_name]
            
            perf['total'] += 1
            if was_correct:
                perf['correct'] += 1
            
            alpha = 0.05
            perf['avg_confidence'] = (1 - alpha) * perf['avg_confidence'] + alpha * confidence
    
    def _default_regime_profiles(self) -> Dict:
        return {
            'trend_up': {
                'cnn': 0.20,
                'lstm': 0.20,
                'orderflow': 0.15,
                'rl': 0.25,
                'meta_reasoner': 0.15,
                'risk': 0.05
            },
            'trend_down': {
                'cnn': 0.20,
                'lstm': 0.20,
                'orderflow': 0.15,
                'rl': 0.25,
                'meta_reasoner': 0.15,
                'risk': 0.05
            },
            'range': {
                'cnn': 0.10,
                'lstm': 0.10,
                'orderflow': 0.30,
                'rl': 0.25,
                'meta_reasoner': 0.15,
                'risk': 0.10
            },
            'high_volatility': {
                'cnn': 0.15,
                'lstm': 0.15,
                'orderflow': 0.20,
                'rl': 0.20,
                'meta_reasoner': 0.10,
                'risk': 0.20
            },
            'low_volatility': {
                'cnn': 0.20,
                'lstm': 0.20,
                'orderflow': 0.15,
                'rl': 0.25,
                'meta_reasoner': 0.15,
                'risk': 0.05
            },
            'choppy': {
                'cnn': 0.10,
                'lstm': 0.10,
                'orderflow': 0.20,
                'rl': 0.20,
                'meta_reasoner': 0.20,
                'risk': 0.20
            },
            'parabolic_run': {
                'cnn': 0.15,
                'lstm': 0.15,
                'orderflow': 0.25,
                'rl': 0.20,
                'meta_reasoner': 0.15,
                'risk': 0.10
            },
            'whale_accumulation': {
                'cnn': 0.10,
                'lstm': 0.10,
                'orderflow': 0.40,
                'rl': 0.20,
                'meta_reasoner': 0.15,
                'risk': 0.05
            },
            'whale_distribution': {
                'cnn': 0.10,
                'lstm': 0.10,
                'orderflow': 0.40,
                'rl': 0.20,
                'meta_reasoner': 0.15,
                'risk': 0.05
            }
        }
    
    def get_fusion_statistics(self) -> Dict:
        if len(self.fusion_history) == 0:
            return {}
        
        recent_actions = [f['action'] for f in list(self.fusion_history)[-100:]]
        recent_confidences = [f['confidence'] for f in list(self.fusion_history)[-100:]]
        
        return {
            'total_fusions': len(self.fusion_history),
            'current_weights': self.signal_weights,
            'signal_performance': self.signal_performance,
            'avg_action': np.mean(recent_actions),
            'std_action': np.std(recent_actions),
            'avg_confidence': np.mean(recent_confidences),
            'last_regime': self.last_regime
        }
