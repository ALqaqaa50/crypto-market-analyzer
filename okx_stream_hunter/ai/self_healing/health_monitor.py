import numpy as np
import logging
from typing import Dict, List
from collections import deque
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(self, performance_window: int = 100, degradation_threshold: float = 0.15,
                 anomaly_threshold: float = 3.0):
        self.performance_window = performance_window
        self.degradation_threshold = degradation_threshold
        self.anomaly_threshold = anomaly_threshold
        
        self.prediction_errors = deque(maxlen=performance_window)
        self.prediction_confidences = deque(maxlen=performance_window)
        self.action_values = deque(maxlen=performance_window)
        
        self.baseline_error = None
        self.baseline_confidence = None
        
        self.health_status = 'healthy'
        self.last_check_time = time.time()
        self.alerts = deque(maxlen=1000)
        
        self.metrics_history = deque(maxlen=5000)
        
        logger.info(f"HealthMonitor initialized: window={performance_window}, "
                   f"degradation_threshold={degradation_threshold}")
    
    def record_prediction(self, predicted_value: float, actual_value: float, 
                         confidence: float = 1.0, action: float = None):
        error = abs(predicted_value - actual_value)
        self.prediction_errors.append(error)
        self.prediction_confidences.append(confidence)
        
        if action is not None:
            self.action_values.append(action)
        
        if self.baseline_error is None and len(self.prediction_errors) >= self.performance_window:
            self.baseline_error = np.mean(list(self.prediction_errors))
            self.baseline_confidence = np.mean(list(self.prediction_confidences))
            logger.info(f"Baseline established: error={self.baseline_error:.4f}, "
                       f"confidence={self.baseline_confidence:.4f}")
    
    def check_model_health(self) -> Dict:
        if len(self.prediction_errors) < self.performance_window * 0.5:
            return {
                'status': 'warming_up',
                'issues': [],
                'metrics': {}
            }
        
        current_error = np.mean(list(self.prediction_errors))
        current_confidence = np.mean(list(self.prediction_confidences))
        
        issues = []
        
        if self.baseline_error is not None:
            error_increase = (current_error - self.baseline_error) / (self.baseline_error + 1e-8)
            
            if error_increase > self.degradation_threshold:
                issues.append({
                    'type': 'performance_degradation',
                    'severity': 'high' if error_increase > 0.3 else 'medium',
                    'metric': 'prediction_error',
                    'baseline': self.baseline_error,
                    'current': current_error,
                    'increase_percent': error_increase * 100
                })
        
        if self.baseline_confidence is not None:
            confidence_drop = (self.baseline_confidence - current_confidence) / (self.baseline_confidence + 1e-8)
            
            if confidence_drop > self.degradation_threshold:
                issues.append({
                    'type': 'confidence_degradation',
                    'severity': 'medium',
                    'metric': 'confidence',
                    'baseline': self.baseline_confidence,
                    'current': current_confidence,
                    'drop_percent': confidence_drop * 100
                })
        
        anomaly_detected = self._detect_anomalies()
        if anomaly_detected:
            issues.append({
                'type': 'prediction_anomaly',
                'severity': 'high',
                'metric': 'prediction_variance',
                'description': 'Unusual prediction patterns detected'
            })
        
        if len(issues) > 0:
            self.health_status = 'degraded'
            for issue in issues:
                self.alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'issue': issue
                })
            logger.warning(f"Model health degraded: {len(issues)} issues detected")
        else:
            self.health_status = 'healthy'
        
        metrics = {
            'current_error': current_error,
            'baseline_error': self.baseline_error,
            'current_confidence': current_confidence,
            'baseline_confidence': self.baseline_confidence,
            'error_std': np.std(list(self.prediction_errors)),
            'confidence_std': np.std(list(self.prediction_confidences)),
            'samples_count': len(self.prediction_errors)
        }
        
        self.metrics_history.append({
            'timestamp': time.time(),
            'status': self.health_status,
            'metrics': metrics
        })
        
        self.last_check_time = time.time()
        
        return {
            'status': self.health_status,
            'issues': issues,
            'metrics': metrics
        }
    
    def _detect_anomalies(self) -> bool:
        if len(self.prediction_errors) < 30:
            return False
        
        recent_errors = list(self.prediction_errors)[-30:]
        
        mean_error = np.mean(recent_errors)
        std_error = np.std(recent_errors)
        
        if std_error < 1e-8:
            return False
        
        for error in recent_errors[-5:]:
            z_score = abs((error - mean_error) / std_error)
            if z_score > self.anomaly_threshold:
                return True
        
        return False
    
    def get_health_report(self) -> Dict:
        if len(self.metrics_history) == 0:
            return {'status': 'no_data'}
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        statuses = [m['status'] for m in recent_metrics]
        status_counts = {
            'healthy': statuses.count('healthy'),
            'degraded': statuses.count('degraded'),
            'warming_up': statuses.count('warming_up')
        }
        
        recent_alerts = list(self.alerts)[-20:]
        
        return {
            'current_status': self.health_status,
            'status_distribution': status_counts,
            'total_checks': len(self.metrics_history),
            'recent_alerts': recent_alerts,
            'last_check': datetime.fromtimestamp(self.last_check_time).isoformat(),
            'baseline_error': self.baseline_error,
            'baseline_confidence': self.baseline_confidence
        }
    
    def reset_baseline(self):
        if len(self.prediction_errors) >= self.performance_window:
            self.baseline_error = np.mean(list(self.prediction_errors))
            self.baseline_confidence = np.mean(list(self.prediction_confidences))
            
            self.health_status = 'healthy'
            
            logger.info(f"Baseline reset: error={self.baseline_error:.4f}, "
                       f"confidence={self.baseline_confidence:.4f}")
