"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Autonomous Intelligence Loop

Continuous learning cycle:
- Periodic model evaluation
- Hyperparameter evolution
- Self-healing triggers
- RL training cycles
- Experience replay optimization
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, Optional
from datetime import datetime

from okx_stream_hunter.ai.brain_ultra import get_brain
from okx_stream_hunter.ai.evolution_engine import GeneticOptimizer, BayesianHyperparameterOptimizer, PopulationBasedTrainer
from okx_stream_hunter.ai.rl_v2.agents.multi_agent import MultiAgentRL
from okx_stream_hunter.notifications import get_telegram_client

logger = logging.getLogger(__name__)


class AutonomousLoop:
    """Supreme autonomous intelligence orchestrator"""
    
    def __init__(self, evaluation_interval: int = 3600, evolution_interval: int = 7200,
                 healing_check_interval: int = 300):
        self.evaluation_interval = evaluation_interval
        self.evolution_interval = evolution_interval
        self.healing_check_interval = healing_check_interval
        
        self.brain = get_brain()
        
        self.telegram = get_telegram_client()
        
        self.is_running = False
        self.loop_thread = None
        self.rl_training_thread = None
        
        self.last_evaluation = 0
        self.last_evolution = 0
        self.last_healing_check = 0
        
        self.cycle_count = 0
        self.performance_history = []
        
        logger.info("AutonomousLoop initialized")
    
    def start(self):
        """Start autonomous intelligence loop"""
        if self.is_running:
            logger.warning("AutonomousLoop already running")
            return
        
        self.is_running = True
        
        self.loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.loop_thread.start()
        
        self.rl_training_thread = threading.Thread(target=self._rl_training_loop, daemon=True)
        self.rl_training_thread.start()
        
        logger.info("🤖 AutonomousLoop STARTED")
    
    def stop(self):
        """Stop autonomous loop"""
        self.is_running = False
        
        if self.loop_thread:
            self.loop_thread.join(timeout=5)
        
        if self.rl_training_thread:
            self.rl_training_thread.join(timeout=5)
        
        logger.info("⏹️ AutonomousLoop STOPPED")
    
    def _run_loop(self):
        """Main autonomous intelligence loop"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_healing_check >= self.healing_check_interval:
                    self._check_health()
                    self.last_healing_check = current_time
                
                if current_time - self.last_evaluation >= self.evaluation_interval:
                    self._evaluate_cycle()
                    self.last_evaluation = current_time
                
                if current_time - self.last_evolution >= self.evolution_interval:
                    self._evolve_hyperparams()
                    self.last_evolution = current_time
                
                time.sleep(10)
            
            except Exception as e:
                logger.error(f"AutonomousLoop error: {e}", exc_info=True)
                time.sleep(60)
    
    def _check_health(self):
        """Check model health and trigger healing if needed"""
        try:
            health_status = self.brain.health_monitor.check_model_health()
            
            if health_status['status'] == 'degraded':
                logger.warning(f"🩺 Health check: Model degraded ({len(health_status['issues'])} issues)")
                
                should_retrain = self.brain.auto_retrainer.register_degradation(health_status)
                
                if should_retrain:
                    logger.critical("🔧 Triggering auto-retraining...")
                    
                    try:
                        self.telegram.send_message_sync(
                            "🔧 <b>[SELF-HEALING]</b> Model degradation detected. Auto-retraining triggered."
                        )
                    except:
                        pass
            else:
                logger.debug(f"✅ Health check: Model healthy")
        
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    def _evaluate_cycle(self):
        """Evaluate current AI performance"""
        try:
            logger.info(f"📊 Evaluation cycle #{self.cycle_count}")
            
            brain_status = self.brain.get_status()
            regime_stats = self.brain.regime_classifier.get_regime_statistics()
            fusion_stats = self.brain.fusion_engine.get_fusion_statistics()
            health_report = self.brain.health_monitor.get_health_report()
            
            performance = {
                'timestamp': time.time(),
                'cycle': self.cycle_count,
                'regime': regime_stats.get('current_regime', 'unknown'),
                'regime_confidence': regime_stats.get('current_confidence', 0.0),
                'fusion_avg_confidence': fusion_stats.get('avg_confidence', 0.0),
                'health_status': health_report.get('current_status', 'unknown'),
                'candles_processed': brain_status['candles_loaded']
            }
            
            self.performance_history.append(performance)
            
            logger.info(f"📊 Cycle #{self.cycle_count}: "
                       f"regime={performance['regime']}, "
                       f"health={performance['health_status']}, "
                       f"fusion_conf={performance['fusion_avg_confidence']:.2%}")
            
            self.cycle_count += 1
        
        except Exception as e:
            logger.error(f"Evaluation cycle error: {e}")
    
    def _evolve_hyperparams(self):
        """Trigger hyperparameter evolution"""
        try:
            logger.info("🧬 Hyperparameter evolution cycle starting...")
            
            logger.info("🧬 Evolution cycle completed (placeholder)")
        
        except Exception as e:
            logger.error(f"Evolution error: {e}")
    
    def _rl_training_loop(self):
        """Continuous RL training thread"""
        training_step = 0
        
        while self.is_running:
            try:
                # Check if any agent has enough samples for training
                can_train = False
                for agent in self.brain.rl_multi_agent.agents:
                    if hasattr(agent, 'replay_buffer') and agent.replay_buffer.size >= 256:
                        can_train = True
                        break
                
                if can_train:
                    self.brain.rl_multi_agent.train(train_all=True)
                    training_step += 1
                    
                    if training_step % 100 == 0:
                        logger.info(f"🎓 RL training step {training_step} completed")
                
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"RL training error: {e}")
                time.sleep(60)
    
    def get_statistics(self) -> Dict:
        """Get autonomous loop statistics"""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'last_evaluation': datetime.fromtimestamp(self.last_evaluation).isoformat() if self.last_evaluation > 0 else None,
            'last_evolution': datetime.fromtimestamp(self.last_evolution).isoformat() if self.last_evolution > 0 else None,
            'last_healing_check': datetime.fromtimestamp(self.last_healing_check).isoformat() if self.last_healing_check > 0 else None,
            'performance_history_length': len(self.performance_history),
            'recent_performance': self.performance_history[-10:] if self.performance_history else []
        }


_autonomous_loop_instance = None

def get_autonomous_loop() -> AutonomousLoop:
    """Get singleton autonomous loop instance"""
    global _autonomous_loop_instance
    if _autonomous_loop_instance is None:
        _autonomous_loop_instance = AutonomousLoop()
    return _autonomous_loop_instance
