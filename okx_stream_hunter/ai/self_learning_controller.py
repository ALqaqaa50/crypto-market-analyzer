"""
PHASE 4: Self-Learning Controller
Safe model promotion and rollback logic
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yaml

from okx_stream_hunter.ai.model_registry import get_model_registry, ModelEntry
from okx_stream_hunter.storage.trade_logger import get_trade_logger
from okx_stream_hunter.backtesting.offline_evaluator import OfflineEvaluator

logger = logging.getLogger(__name__)


class SelfLearningController:
    """
    Controls safe model training and promotion
    Enforces safety rules and prevents production degradation
    """
    
    def __init__(self, config_path: str = "okx_stream_hunter/config/trading_config.yaml"):
        self.config = self._load_config(config_path)
        self.registry = get_model_registry()
        self.trade_logger = get_trade_logger()
        
        self.sl_config = self.config.get('self_learning', {})
        
        logger.info("🎓 SelfLearningController initialized")
    
    def is_self_learning_enabled(self) -> bool:
        """Check if self-learning is enabled"""
        return self.sl_config.get('enable_self_learning', False)
    
    def is_shadow_mode_enabled(self) -> bool:
        """Check if shadow mode is enabled"""
        return self.sl_config.get('enable_shadow_mode', False)
    
    def should_trigger_training(self, model_type: str = "cnn") -> bool:
        """Check if training should be triggered"""
        
        if not self.is_self_learning_enabled():
            return False
        
        try:
            # Get trade count
            stats = self.trade_logger.get_stats()
            total_logged = stats.get('total_logged', 0)
            
            min_trades = self.sl_config.get('min_trades_before_retrain', 100)
            
            if total_logged < min_trades:
                logger.debug(f"Not enough trades: {total_logged}/{min_trades}")
                return False
            
            # Check last training time
            current_prod = self.registry.get_current_production_model(model_type)
            if current_prod:
                promoted_at = datetime.fromisoformat(current_prod.promoted_at)
                interval_hours = self.sl_config.get('retrain_interval_hours', 24)
                next_training = promoted_at + timedelta(hours=interval_hours)
                
                if datetime.utcnow() < next_training:
                    logger.debug(f"Too soon to retrain (next: {next_training})")
                    return False
            
            logger.info(f"✅ Training should be triggered (trades={total_logged})")
            return True
        
        except Exception as e:
            logger.error(f"❌ Error checking training trigger: {e}")
            return False
    
    def check_promotion_criteria(
        self,
        candidate_metrics: Dict,
        production_metrics: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Check if candidate meets promotion criteria
        Returns (should_promote, reason)
        """
        
        try:
            # Get thresholds
            min_sharpe = self.sl_config.get('min_eval_sharpe_for_promotion', 1.0)
            min_winrate = self.sl_config.get('min_eval_winrate_for_promotion', 55.0)
            max_dd = self.sl_config.get('max_allowed_drawdown_for_candidate', 20.0)
            min_improvement = self.sl_config.get('min_improvement_over_production', 0.05)
            
            # Check absolute thresholds
            sharpe = candidate_metrics.get('sharpe_ratio', 0)
            winrate = candidate_metrics.get('win_rate', 0)
            drawdown = abs(candidate_metrics.get('max_drawdown', 0))
            
            if sharpe < min_sharpe:
                return False, f"Sharpe too low: {sharpe:.2f} < {min_sharpe}"
            
            if winrate < min_winrate:
                return False, f"Win rate too low: {winrate:.1f}% < {min_winrate}%"
            
            if drawdown > max_dd:
                return False, f"Drawdown too high: {drawdown:.1f}% > {max_dd}%"
            
            # Check improvement over production
            if production_metrics:
                prod_sharpe = production_metrics.get('sharpe_ratio', 0)
                improvement = (sharpe - prod_sharpe) / max(prod_sharpe, 0.1)
                
                if improvement < min_improvement:
                    return False, f"Insufficient improvement: {improvement*100:.1f}% < {min_improvement*100:.1f}%"
            
            # Check total trades
            total_trades = candidate_metrics.get('total_trades', 0)
            if total_trades < 30:
                return False, f"Not enough trades in backtest: {total_trades} < 30"
            
            return True, "All criteria met"
        
        except Exception as e:
            logger.error(f"❌ Error checking criteria: {e}")
            return False, f"Error: {e}"
    
    def evaluate_and_promote(
        self,
        model_type: str,
        candidate_version: str,
        test_data: Optional[object] = None,
        auto_promote: bool = False
    ) -> bool:
        """
        Evaluate candidate and promote if criteria met
        Returns True if promoted
        """
        
        try:
            if not self.is_self_learning_enabled():
                logger.warning("Self-learning not enabled")
                return False
            
            # Load candidate model
            candidate = None
            for model in self.registry.get_all_models(model_type):
                if model.version_id == candidate_version:
                    candidate = model
                    break
            
            if not candidate:
                logger.error(f"Candidate not found: {candidate_version}")
                return False
            
            # Get production model metrics
            prod_model = self.registry.get_current_production_model(model_type)
            prod_metrics = prod_model.metrics if prod_model else None
            
            # Get candidate metrics
            candidate_metrics = candidate.metrics
            
            # If test_data provided, run full evaluation
            if test_data is not None:
                logger.info(f"📊 Evaluating candidate on test data...")
                evaluator = OfflineEvaluator()
                
                # Load model
                from okx_stream_hunter.ai.offline_trainer import OfflineTrainer
                trainer = OfflineTrainer(model_type=model_type)
                model = trainer.load_model(candidate.file_path)
                
                # Evaluate
                eval_metrics = evaluator.evaluate_model(model, test_data, model_type)
                
                # Update candidate metrics
                candidate_metrics.update(eval_metrics)
                candidate.metrics = candidate_metrics
                self.registry._save_registry()
            
            # Check promotion criteria
            should_promote, reason = self.check_promotion_criteria(
                candidate_metrics,
                prod_metrics
            )
            
            if not should_promote:
                logger.warning(f"❌ Candidate does not meet criteria: {reason}")
                return False
            
            logger.info(f"✅ Candidate meets criteria: {reason}")
            
            # Check manual approval requirement
            manual_approval = self.sl_config.get('manual_approval_required', True)
            
            if manual_approval and not auto_promote:
                logger.warning("⚠️ Manual approval required for promotion")
                logger.info(f"To promote manually, run:")
                logger.info(f"  registry.promote_to_production('{model_type}', '{candidate_version}')")
                return False
            
            # Promote
            success = self.registry.promote_to_production(model_type, candidate_version)
            
            if success:
                logger.info(f"🚀 Promoted {candidate_version} to production")
            else:
                logger.error(f"❌ Failed to promote {candidate_version}")
            
            return success
        
        except Exception as e:
            logger.error(f"❌ Evaluate and promote error: {e}")
            return False
    
    def check_production_performance(
        self,
        model_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Monitor production model performance
        Returns (is_degraded, reason)
        """
        
        try:
            if not self.is_self_learning_enabled():
                return False, None
            
            if not self.sl_config.get('allow_auto_rollback', False):
                return False, None
            
            prod_model = self.registry.get_current_production_model(model_type)
            if not prod_model:
                return False, None
            
            # Get recent trades
            window_trades = self.sl_config.get('performance_monitor_window_trades', 50)
            
            # Load recent data
            data = self.trade_logger.load_data(days_back=7)
            if data is None or len(data) < window_trades:
                return False, None
            
            recent_data = data.tail(window_trades)
            
            # Calculate recent win rate
            if 'trade_pnl' in recent_data.columns:
                winning_trades = (recent_data['trade_pnl'] > 0).sum()
                recent_winrate = (winning_trades / len(recent_data)) * 100
            else:
                return False, None
            
            # Get original win rate
            original_winrate = prod_model.metrics.get('win_rate', 50)
            
            # Check degradation threshold
            winrate_drop = original_winrate - recent_winrate
            threshold = self.sl_config.get('rollback_threshold_winrate_drop', 10.0)
            
            if winrate_drop > threshold:
                return True, f"Win rate dropped {winrate_drop:.1f}% (from {original_winrate:.1f}% to {recent_winrate:.1f}%)"
            
            return False, None
        
        except Exception as e:
            logger.error(f"❌ Performance check error: {e}")
            return False, None
    
    def trigger_rollback(self, model_type: str) -> bool:
        """Rollback production model to previous version"""
        
        try:
            if not self.sl_config.get('allow_auto_rollback', False):
                logger.warning("Auto-rollback not enabled")
                return False
            
            success = self.registry.rollback_to_previous(model_type)
            
            if success:
                logger.info(f"⏪ Rolled back {model_type} to previous version")
            else:
                logger.error(f"❌ Rollback failed for {model_type}")
            
            return success
        
        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False
    
    def get_learning_status(self) -> Dict:
        """Get current self-learning status"""
        
        stats = {
            'enabled': self.is_self_learning_enabled(),
            'shadow_mode': self.is_shadow_mode_enabled(),
            'config': {
                'min_trades_before_retrain': self.sl_config.get('min_trades_before_retrain'),
                'min_eval_sharpe_for_promotion': self.sl_config.get('min_eval_sharpe_for_promotion'),
                'manual_approval_required': self.sl_config.get('manual_approval_required'),
                'allow_auto_rollback': self.sl_config.get('allow_auto_rollback')
            },
            'data': {
                'total_logged_trades': self.trade_logger.get_stats().get('total_logged', 0),
                'last_flush': self.trade_logger.get_stats().get('last_flush')
            },
            'registry': self.registry.get_registry_stats(),
            'production_models': {},
            'best_candidates': {}
        }
        
        # Get production models
        for model_type in ['cnn', 'lstm', 'rl_policy', 'rl_value']:
            prod = self.registry.get_current_production_model(model_type)
            if prod:
                stats['production_models'][model_type] = {
                    'version': prod.version_id,
                    'promoted_at': prod.promoted_at,
                    'metrics': prod.metrics
                }
            
            # Get best candidate
            candidate = self.registry.get_best_candidate(model_type, 'test_accuracy')
            if candidate:
                stats['best_candidates'][model_type] = {
                    'version': candidate.version_id,
                    'registered_at': candidate.registered_at,
                    'metrics': candidate.metrics
                }
        
        return stats
    
    def _load_config(self, path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}


# Global instance
_controller_instance: Optional[SelfLearningController] = None


def get_self_learning_controller(config_path: str = "okx_stream_hunter/config/trading_config.yaml") -> SelfLearningController:
    """Get global controller instance"""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = SelfLearningController(config_path)
    return _controller_instance
