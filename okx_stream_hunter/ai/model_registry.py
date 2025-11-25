"""
PHASE 4: Model Registry
Version control and management for AI models
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Model registry entry"""
    version_id: str
    model_type: str  # cnn, lstm, rl_policy, rl_value, fusion
    file_path: str
    status: str  # production, candidate, archived
    training_config_hash: str
    metrics: Dict
    registered_at: str
    promoted_at: Optional[str] = None
    archived_at: Optional[str] = None
    previous_version: Optional[str] = None


class ModelRegistry:
    """
    Central registry for model version control
    Manages production, candidate, and archived models
    """
    
    def __init__(self, registry_file: str = "storage/model_registry.json"):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, List[ModelEntry]] = {}
        
        self._load_registry()
        
        logger.info(f"📚 ModelRegistry initialized (file={registry_file})")
    
    def register_model(
        self,
        version_id: str,
        model_type: str,
        file_path: str,
        training_config: Dict,
        metrics: Dict,
        status: str = "candidate"
    ) -> bool:
        """Register a new model"""
        
        try:
            config_hash = self._hash_config(training_config)
            
            entry = ModelEntry(
                version_id=version_id,
                model_type=model_type,
                file_path=file_path,
                status=status,
                training_config_hash=config_hash,
                metrics=metrics,
                registered_at=datetime.utcnow().isoformat()
            )
            
            if model_type not in self.models:
                self.models[model_type] = []
            
            self.models[model_type].append(entry)
            self._save_registry()
            
            logger.info(f"✅ Model registered: {model_type}/{version_id} ({status})")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Register model error: {e}")
            return False
    
    def get_current_production_model(self, model_type: str) -> Optional[ModelEntry]:
        """Get current production model"""
        
        if model_type not in self.models:
            return None
        
        production_models = [m for m in self.models[model_type] if m.status == "production"]
        
        if not production_models:
            return None
        
        # Return most recently promoted
        production_models.sort(key=lambda m: m.promoted_at or m.registered_at, reverse=True)
        
        return production_models[0]
    
    def get_best_candidate(
        self,
        model_type: str,
        metric_name: str = "test_accuracy"
    ) -> Optional[ModelEntry]:
        """Get best candidate model by metric"""
        
        if model_type not in self.models:
            return None
        
        candidates = [m for m in self.models[model_type] if m.status == "candidate"]
        
        if not candidates:
            return None
        
        # Sort by metric (descending)
        candidates.sort(
            key=lambda m: m.metrics.get(metric_name, 0),
            reverse=True
        )
        
        return candidates[0]
    
    def get_all_models(self, model_type: str) -> List[ModelEntry]:
        """Get all models of a type"""
        return self.models.get(model_type, [])
    
    def promote_to_production(
        self,
        model_type: str,
        version_id: str,
        keep_previous: bool = True
    ) -> bool:
        """Promote candidate to production"""
        
        try:
            if model_type not in self.models:
                logger.error(f"Model type not found: {model_type}")
                return False
            
            # Find candidate
            candidate = None
            for model in self.models[model_type]:
                if model.version_id == version_id:
                    candidate = model
                    break
            
            if not candidate:
                logger.error(f"Model not found: {version_id}")
                return False
            
            if candidate.status != "candidate":
                logger.warning(f"Model {version_id} is not a candidate (status={candidate.status})")
            
            # Get current production model
            current_prod = self.get_current_production_model(model_type)
            
            # Archive current production if exists
            if current_prod:
                if keep_previous:
                    current_prod.status = "archived"
                    current_prod.archived_at = datetime.utcnow().isoformat()
                    candidate.previous_version = current_prod.version_id
                    logger.info(f"📦 Archived previous production: {current_prod.version_id}")
            
            # Promote candidate
            candidate.status = "production"
            candidate.promoted_at = datetime.utcnow().isoformat()
            
            self._save_registry()
            
            logger.info(f"🚀 Promoted to production: {model_type}/{version_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Promote error: {e}")
            return False
    
    def archive_model(self, model_type: str, version_id: str) -> bool:
        """Archive a model"""
        
        try:
            if model_type not in self.models:
                return False
            
            for model in self.models[model_type]:
                if model.version_id == version_id:
                    model.status = "archived"
                    model.archived_at = datetime.utcnow().isoformat()
                    self._save_registry()
                    
                    logger.info(f"📦 Archived model: {model_type}/{version_id}")
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"❌ Archive error: {e}")
            return False
    
    def rollback_to_previous(self, model_type: str) -> bool:
        """Rollback production to previous version"""
        
        try:
            current_prod = self.get_current_production_model(model_type)
            
            if not current_prod or not current_prod.previous_version:
                logger.warning(f"No previous version to rollback to")
                return False
            
            previous_id = current_prod.previous_version
            
            # Find previous model
            previous_model = None
            for model in self.models[model_type]:
                if model.version_id == previous_id:
                    previous_model = model
                    break
            
            if not previous_model:
                logger.error(f"Previous model not found: {previous_id}")
                return False
            
            # Swap statuses
            current_prod.status = "archived"
            current_prod.archived_at = datetime.utcnow().isoformat()
            
            previous_model.status = "production"
            previous_model.promoted_at = datetime.utcnow().isoformat()
            
            self._save_registry()
            
            logger.info(f"⏪ Rolled back to previous version: {previous_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Rollback error: {e}")
            return False
    
    def get_registry_stats(self) -> Dict:
        """Get registry statistics"""
        stats = {
            'total_models': sum(len(models) for models in self.models.values()),
            'by_type': {},
            'by_status': {
                'production': 0,
                'candidate': 0,
                'archived': 0
            }
        }
        
        for model_type, models in self.models.items():
            stats['by_type'][model_type] = {
                'total': len(models),
                'production': len([m for m in models if m.status == 'production']),
                'candidate': len([m for m in models if m.status == 'candidate']),
                'archived': len([m for m in models if m.status == 'archived'])
            }
            
            for model in models:
                stats['by_status'][model.status] += 1
        
        return stats
    
    def _hash_config(self, config: Dict) -> str:
        """Generate hash of training configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _load_registry(self):
        """Load registry from disk"""
        if not self.registry_file.exists():
            logger.info("📝 Creating new registry file")
            self._save_registry()
            return
        
        try:
            data = json.loads(self.registry_file.read_text())
            
            for model_type, entries in data.items():
                self.models[model_type] = [
                    ModelEntry(**entry) for entry in entries
                ]
            
            logger.info(f"✅ Registry loaded: {len(self.models)} model types")
        
        except Exception as e:
            logger.error(f"❌ Load registry error: {e}")
            self.models = {}
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {
                model_type: [asdict(entry) for entry in entries]
                for model_type, entries in self.models.items()
            }
            
            self.registry_file.write_text(json.dumps(data, indent=2))
            
            logger.debug(f"💾 Registry saved")
        
        except Exception as e:
            logger.error(f"❌ Save registry error: {e}")


# Global instance
_registry_instance: Optional[ModelRegistry] = None


def get_model_registry(registry_file: str = "storage/model_registry.json") -> ModelRegistry:
    """Get global registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(registry_file)
    return _registry_instance


def reset_model_registry(registry_file: str = "storage/model_registry.json"):
    """Reset global registry (for testing)"""
    global _registry_instance
    _registry_instance = ModelRegistry(registry_file)
