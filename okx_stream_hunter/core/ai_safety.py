"""
AI Safety Layer - PHASE 3
Anomaly detection, confidence validation, emergency stop mechanisms
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class AISafetyLayer:
    """AI safety and anomaly detection system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.decision_history = deque(maxlen=500)
        self.confidence_history = deque(maxlen=200)
        self.pnl_history = deque(maxlen=200)
        
        self.anomaly_count = 0
        self.emergency_stops = 0
        self.safety_interventions = 0
        
        self.is_emergency_stopped = False
        self.emergency_reason = None
        
        self.confidence_floor = config.get('confidence_floor', 0.3)
        self.confidence_ceiling = config.get('confidence_ceiling', 0.95)
        self.max_confidence_std = config.get('max_confidence_std', 0.3)
        
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        self.max_drawdown_pct = config.get('max_drawdown_pct', 0.15)
        
        self.min_decision_interval = config.get('min_decision_interval_ms', 100)
        self.last_decision_time = None
        
        logger.info("🛡️ AI Safety Layer initialized")
    
    async def validate_decision(self, decision: Dict) -> Dict:
        """Comprehensive safety validation of AI decision"""
        try:
            validation = {
                'safe': True,
                'warnings': [],
                'blocks': [],
                'confidence_adjusted': False
            }
            
            if self.is_emergency_stopped:
                validation['safe'] = False
                validation['blocks'].append(f'Emergency stop active: {self.emergency_reason}')
                return validation
            
            confidence = decision.get('confidence', 0.0)
            direction = decision.get('direction', 'NEUTRAL')
            
            anomaly = await self._detect_anomaly(decision)
            if anomaly['detected']:
                self.anomaly_count += 1
                validation['warnings'].append(f"Anomaly: {anomaly['type']}")
                
                if anomaly['severity'] == 'critical':
                    validation['safe'] = False
                    validation['blocks'].append(anomaly['reason'])
                    self.safety_interventions += 1
            
            if confidence < self.confidence_floor:
                validation['warnings'].append(f'Confidence below floor: {confidence:.2%}')
                decision['confidence'] = self.confidence_floor
                validation['confidence_adjusted'] = True
            
            if confidence > self.confidence_ceiling:
                validation['warnings'].append(f'Confidence above ceiling: {confidence:.2%}')
                decision['confidence'] = self.confidence_ceiling
                validation['confidence_adjusted'] = True
            
            confidence_check = self._check_confidence_stability()
            if not confidence_check['stable']:
                validation['warnings'].append('Confidence unstable')
                if confidence_check['severity'] == 'high':
                    validation['safe'] = False
                    validation['blocks'].append('Confidence volatility too high')
            
            if self._check_decision_rate_limit():
                validation['safe'] = False
                validation['blocks'].append('Decision rate limit exceeded')
            
            consecutive_losses = self._count_consecutive_losses()
            if consecutive_losses >= self.max_consecutive_losses:
                validation['safe'] = False
                validation['blocks'].append(f'{consecutive_losses} consecutive losses detected')
                await self._trigger_emergency_stop('consecutive_losses')
            
            drawdown = self._calculate_current_drawdown()
            if drawdown >= self.max_drawdown_pct:
                validation['safe'] = False
                validation['blocks'].append(f'Max drawdown exceeded: {drawdown:.2%}')
                await self._trigger_emergency_stop('max_drawdown')
            
            pattern_check = self._detect_suspicious_pattern()
            if pattern_check['suspicious']:
                validation['warnings'].append(f'Suspicious pattern: {pattern_check["type"]}')
                if pattern_check['severity'] == 'high':
                    validation['safe'] = False
                    validation['blocks'].append(pattern_check['reason'])
            
            self.decision_history.append({
                'timestamp': datetime.now(),
                'decision': direction,
                'confidence': confidence,
                'safe': validation['safe']
            })
            
            self.confidence_history.append(confidence)
            self.last_decision_time = datetime.now()
            
            if not validation['safe']:
                logger.warning(f"⚠️ Decision blocked by safety layer: {validation['blocks']}")
            elif validation['warnings']:
                logger.info(f"⚡ Safety warnings: {validation['warnings']}")
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ Safety validation error: {e}")
            return {
                'safe': False,
                'warnings': [],
                'blocks': [f'Safety system error: {str(e)}'],
                'confidence_adjusted': False
            }
    
    async def _detect_anomaly(self, decision: Dict) -> Dict:
        """Detect anomalies in AI decision"""
        anomaly = {
            'detected': False,
            'type': None,
            'severity': 'low',
            'reason': ''
        }
        
        confidence = decision.get('confidence', 0.0)
        direction = decision.get('direction', 'NEUTRAL')
        
        if confidence == 0.0 or confidence == 1.0:
            anomaly['detected'] = True
            anomaly['type'] = 'extreme_confidence'
            anomaly['severity'] = 'high'
            anomaly['reason'] = f'Extreme confidence value: {confidence}'
            return anomaly
        
        if len(self.decision_history) >= 10:
            recent_directions = [d['decision'] for d in list(self.decision_history)[-10:]]
            if direction != 'NEUTRAL' and recent_directions.count(direction) == 0:
                anomaly['detected'] = True
                anomaly['type'] = 'sudden_direction_change'
                anomaly['severity'] = 'medium'
                anomaly['reason'] = f'Sudden change to {direction} after {recent_directions[-5:]}'
        
        if len(self.confidence_history) >= 20:
            recent_conf = list(self.confidence_history)[-20:]
            conf_std = np.std(recent_conf)
            if conf_std > self.max_confidence_std:
                anomaly['detected'] = True
                anomaly['type'] = 'high_confidence_volatility'
                anomaly['severity'] = 'high'
                anomaly['reason'] = f'Confidence std: {conf_std:.3f}'
        
        features = decision.get('features', {})
        if not features or len(features) < 3:
            anomaly['detected'] = True
            anomaly['type'] = 'missing_features'
            anomaly['severity'] = 'critical'
            anomaly['reason'] = 'Decision missing critical features'
        
        return anomaly
    
    def _check_confidence_stability(self) -> Dict:
        """Check if confidence values are stable"""
        if len(self.confidence_history) < 10:
            return {'stable': True, 'severity': 'low'}
        
        recent = list(self.confidence_history)[-10:]
        mean_conf = np.mean(recent)
        std_conf = np.std(recent)
        
        if std_conf > 0.25:
            return {'stable': False, 'severity': 'high', 'std': std_conf}
        elif std_conf > 0.15:
            return {'stable': False, 'severity': 'medium', 'std': std_conf}
        
        return {'stable': True, 'severity': 'low', 'std': std_conf}
    
    def _check_decision_rate_limit(self) -> bool:
        """Check if decisions are being made too rapidly"""
        if self.last_decision_time is None:
            return False
        
        elapsed_ms = (datetime.now() - self.last_decision_time).total_seconds() * 1000
        return elapsed_ms < self.min_decision_interval
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losing trades"""
        if not self.pnl_history:
            return 0
        
        consecutive = 0
        for pnl in reversed(list(self.pnl_history)):
            if pnl < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.pnl_history:
            return 0.0
        
        cumulative_pnl = np.cumsum([p for p in self.pnl_history])
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / (peak + 1.0)
        
        return float(drawdown[-1]) if len(drawdown) > 0 else 0.0
    
    def _detect_suspicious_pattern(self) -> Dict:
        """Detect suspicious decision patterns"""
        if len(self.decision_history) < 20:
            return {'suspicious': False, 'type': None, 'severity': 'low', 'reason': ''}
        
        recent = list(self.decision_history)[-20:]
        
        same_decision_count = {}
        for d in recent:
            decision = d['decision']
            same_decision_count[decision] = same_decision_count.get(decision, 0) + 1
        
        for decision, count in same_decision_count.items():
            if count >= 15 and decision != 'NEUTRAL':
                return {
                    'suspicious': True,
                    'type': 'repetitive_decisions',
                    'severity': 'high',
                    'reason': f'{decision} repeated {count}/20 times'
                }
        
        confidences = [d['confidence'] for d in recent]
        if all(c > 0.9 for c in confidences[-5:]):
            return {
                'suspicious': True,
                'type': 'unrealistic_confidence',
                'severity': 'high',
                'reason': 'All recent decisions >90% confidence'
            }
        
        return {'suspicious': False, 'type': None, 'severity': 'low', 'reason': ''}
    
    async def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.is_emergency_stopped = True
        self.emergency_reason = reason
        self.emergency_stops += 1
        
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
    
    def reset_emergency_stop(self):
        """Manually reset emergency stop"""
        self.is_emergency_stopped = False
        self.emergency_reason = None
        logger.info("✅ Emergency stop reset")
    
    def record_trade_outcome(self, pnl: float):
        """Record trade outcome for safety monitoring"""
        self.pnl_history.append(pnl)
    
    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        return {
            'emergency_stopped': self.is_emergency_stopped,
            'emergency_reason': self.emergency_reason,
            'anomaly_count': self.anomaly_count,
            'safety_interventions': self.safety_interventions,
            'consecutive_losses': self._count_consecutive_losses(),
            'current_drawdown': self._calculate_current_drawdown(),
            'confidence_stability': self._check_confidence_stability(),
            'total_decisions_monitored': len(self.decision_history)
        }
    
    def get_health_score(self) -> float:
        """Calculate overall safety health score (0-1)"""
        if self.is_emergency_stopped:
            return 0.0
        
        score = 1.0
        
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses > 0:
            score -= (consecutive_losses / self.max_consecutive_losses) * 0.3
        
        drawdown = self._calculate_current_drawdown()
        score -= (drawdown / self.max_drawdown_pct) * 0.3
        
        if len(self.confidence_history) >= 10:
            conf_std = np.std(list(self.confidence_history)[-10:])
            score -= (conf_std / self.max_confidence_std) * 0.2
        
        if self.anomaly_count > 0:
            anomaly_rate = self.anomaly_count / max(len(self.decision_history), 1)
            score -= anomaly_rate * 0.2
        
        return max(0.0, min(1.0, score))
