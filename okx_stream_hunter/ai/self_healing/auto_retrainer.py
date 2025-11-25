import numpy as np
import logging
import threading
import time
from typing import Dict, Callable
from queue import Queue
import copy

from okx_stream_hunter.notifications import get_telegram_client

logger = logging.getLogger(__name__)


class AutoRetrainer:
    def __init__(self, retrain_threshold: int = 3, cooldown_period: int = 3600,
                 validation_threshold: float = 0.05):
        self.retrain_threshold = retrain_threshold
        self.cooldown_period = cooldown_period
        self.validation_threshold = validation_threshold
        
        self.degradation_count = 0
        self.last_retrain_time = 0
        
        self.is_retraining = False
        self.retrain_thread = None
        self.retrain_queue = Queue()
        
        self.active_model = None
        self.backup_model = None
        self.candidate_model = None
        
        self.retrain_history = []
        
        self.model_lock = threading.Lock()
        
        self.telegram = get_telegram_client()
        
        logger.info(f"AutoRetrainer initialized: threshold={retrain_threshold}, "
                   f"cooldown={cooldown_period}s")
    
    def register_degradation(self, health_status: Dict):
        if health_status['status'] == 'degraded':
            self.degradation_count += 1
            logger.info(f"Degradation registered: count={self.degradation_count}/{self.retrain_threshold}")
            
            if self.degradation_count >= self.retrain_threshold:
                time_since_last_retrain = time.time() - self.last_retrain_time
                
                if time_since_last_retrain >= self.cooldown_period:
                    logger.warning(f"Triggering retraining after {self.degradation_count} degradations")
                    return True
                else:
                    logger.info(f"Retraining on cooldown: {int(self.cooldown_period - time_since_last_retrain)}s remaining")
                    return False
        else:
            if self.degradation_count > 0:
                self.degradation_count = max(0, self.degradation_count - 1)
        
        return False
    
    def trigger_retrain(self, model_generator: Callable, training_data: Dict,
                       validation_data: Dict = None) -> bool:
        if self.is_retraining:
            logger.warning("Retraining already in progress, skipping")
            return False
        
        self.is_retraining = True
        
        self.retrain_thread = threading.Thread(
            target=self._retrain_worker,
            args=(model_generator, training_data, validation_data),
            daemon=True
        )
        self.retrain_thread.start()
        
        return True
    
    def _retrain_worker(self, model_generator: Callable, training_data: Dict,
                       validation_data: Dict = None):
        try:
            logger.info("Background retraining started")
            retrain_start = time.time()
            
            with self.model_lock:
                if self.active_model is not None:
                    self.backup_model = copy.deepcopy(self.active_model)
            
            self.candidate_model = model_generator()
            
            logger.info("Training candidate model...")
            self.candidate_model.fit(training_data)
            
            if validation_data is not None:
                validation_performance = self._validate_candidate(validation_data)
                
                if validation_performance['passed']:
                    self._swap_model()
                    logger.info(f"Model swap successful: validation_score={validation_performance['score']:.4f}")
                    
                    try:
                        self.telegram.send_message_sync(
                            f"🔄 <b>[MODEL UPDATE]</b> Auto-retraining completed successfully. "
                            f"Validation score: {validation_performance['score']:.4f}"
                        )
                    except:
                        pass
                else:
                    logger.warning(f"Candidate model failed validation: "
                                 f"score={validation_performance['score']:.4f}, "
                                 f"threshold={self.validation_threshold}")
                    self.candidate_model = None
            else:
                self._swap_model()
                logger.info("Model swap completed (no validation)")
            
            retrain_duration = time.time() - retrain_start
            
            self.retrain_history.append({
                'timestamp': time.time(),
                'duration': retrain_duration,
                'success': validation_data is None or validation_performance['passed']
            })
            
            self.last_retrain_time = time.time()
            self.degradation_count = 0
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            self.retrain_history.append({
                'timestamp': time.time(),
                'duration': time.time() - retrain_start,
                'success': False,
                'error': str(e)
            })
        finally:
            self.is_retraining = False
    
    def _validate_candidate(self, validation_data: Dict) -> Dict:
        try:
            candidate_predictions = self.candidate_model.predict(validation_data['X'])
            candidate_error = np.mean(np.abs(candidate_predictions - validation_data['y']))
            
            if self.active_model is not None:
                active_predictions = self.active_model.predict(validation_data['X'])
                active_error = np.mean(np.abs(active_predictions - validation_data['y']))
                
                improvement = (active_error - candidate_error) / (active_error + 1e-8)
                
                passed = improvement > -self.validation_threshold
                
                return {
                    'passed': passed,
                    'score': improvement,
                    'candidate_error': candidate_error,
                    'active_error': active_error
                }
            else:
                return {
                    'passed': True,
                    'score': 1.0,
                    'candidate_error': candidate_error,
                    'active_error': None
                }
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'passed': False,
                'score': -1.0,
                'error': str(e)
            }
    
    def _swap_model(self):
        with self.model_lock:
            if self.active_model is not None:
                self.backup_model = self.active_model
            
            self.active_model = self.candidate_model
            self.candidate_model = None
        
        logger.info("Model swap completed")
    
    def get_active_model(self):
        with self.model_lock:
            return self.active_model
    
    def rollback(self) -> bool:
        with self.model_lock:
            if self.backup_model is not None:
                logger.warning("Rolling back to backup model")
                self.active_model = self.backup_model
                self.backup_model = None
                return True
            else:
                logger.error("No backup model available for rollback")
                return False
    
    def get_retrain_statistics(self) -> Dict:
        successful_retrains = sum(1 for r in self.retrain_history if r['success'])
        total_retrains = len(self.retrain_history)
        
        return {
            'total_retrains': total_retrains,
            'successful_retrains': successful_retrains,
            'is_retraining': self.is_retraining,
            'degradation_count': self.degradation_count,
            'last_retrain_time': self.last_retrain_time,
            'has_backup': self.backup_model is not None,
            'retrain_history': self.retrain_history[-10:]
        }
