"""
PHASE 4: Offline Trainer
Train AI models offline without impacting live trading
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Detect deep learning framework
try:
    import tensorflow as tf
    from tensorflow import keras
    FRAMEWORK = "tensorflow"
    logger.info("🔥 Using TensorFlow for training")
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        FRAMEWORK = "pytorch"
        logger.info("🔥 Using PyTorch for training")
    except ImportError:
        FRAMEWORK = None
        logger.warning("⚠️ No deep learning framework found (TensorFlow/PyTorch)")


class OfflineTrainer:
    """
    Offline model trainer for CNN, LSTM, and RL components
    Does not interfere with live trading
    """
    
    def __init__(
        self,
        model_type: str = "cnn",  # cnn, lstm, rl_policy, rl_value
        config: Dict = None
    ):
        self.model_type = model_type
        self.config = config or {}
        self.framework = FRAMEWORK
        
        self.model = None
        self.training_history = []
        
        logger.info(f"🎓 OfflineTrainer initialized for {model_type} using {self.framework}")
    
    def build_model(self, input_shape: Tuple, num_classes: int = 2) -> bool:
        """Build model architecture"""
        try:
            if self.framework == "tensorflow":
                self.model = self._build_tensorflow_model(input_shape, num_classes)
            elif self.framework == "pytorch":
                self.model = self._build_pytorch_model(input_shape, num_classes)
            else:
                logger.error("No framework available")
                return False
            
            logger.info(f"✅ Model built: input_shape={input_shape}, output={num_classes}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Model build error: {e}")
            return False
    
    def _build_tensorflow_model(self, input_shape: Tuple, num_classes: int):
        """Build TensorFlow/Keras model"""
        
        if self.model_type == "cnn":
            model = keras.Sequential([
                keras.layers.Conv1D(32, 3, activation='relu', input_shape=input_shape[1:]),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(64, 3, activation='relu'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(128, 3, activation='relu'),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
            ])
        
        elif self.model_type == "lstm":
            model = keras.Sequential([
                keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape[1:]),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
            ])
        
        else:  # Generic dense
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=input_shape[1:]),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
            ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def _build_pytorch_model(self, input_shape: Tuple, num_classes: int):
        """Build PyTorch model (simplified)"""
        logger.warning("⚠️ PyTorch model building not fully implemented")
        return None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict:
        """Train model"""
        
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return {}
        
        if X_train.size == 0 or y_train.size == 0:
            logger.error("Empty training data")
            return {}
        
        try:
            logger.info(f"🎓 Training {self.model_type} model...")
            logger.info(f"   Train samples: {len(X_train)}")
            logger.info(f"   Val samples: {len(X_val) if X_val is not None else 0}")
            logger.info(f"   Epochs: {epochs}, Batch size: {batch_size}")
            
            start_time = datetime.utcnow()
            
            if self.framework == "tensorflow":
                validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
                
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=validation_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1,
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor='val_loss' if validation_data else 'loss',
                            patience=10,
                            restore_best_weights=True
                        )
                    ]
                )
                
                metrics = {
                    'train_loss': float(history.history['loss'][-1]),
                    'train_accuracy': float(history.history['accuracy'][-1]),
                }
                
                if validation_data:
                    metrics['val_loss'] = float(history.history['val_loss'][-1])
                    metrics['val_accuracy'] = float(history.history['val_accuracy'][-1])
            
            else:
                logger.warning("Training not implemented for current framework")
                metrics = {}
            
            end_time = datetime.utcnow()
            training_duration = (end_time - start_time).total_seconds()
            
            metrics['training_duration_seconds'] = training_duration
            metrics['timestamp'] = end_time.isoformat()
            
            self.training_history.append(metrics)
            
            logger.info(f"✅ Training complete in {training_duration:.1f}s")
            logger.info(f"   Final metrics: {metrics}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Training error: {e}", exc_info=True)
            return {}
    
    def validate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Validate model on test set"""
        
        if self.model is None:
            logger.error("Model not available")
            return {}
        
        if X_test.size == 0 or y_test.size == 0:
            logger.error("Empty test data")
            return {}
        
        try:
            logger.info(f"📊 Validating on {len(X_test)} samples...")
            
            if self.framework == "tensorflow":
                loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
                
                y_pred_proba = self.model.predict(X_test, verbose=0)
                y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.shape[1] > 1 else (y_pred_proba > 0.5).astype(int).flatten()
                
                # Compute additional metrics
                from sklearn.metrics import precision_score, recall_score, f1_score
                
                precision = precision_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='binary' if len(np.unique(y_test)) == 2 else 'macro', zero_division=0)
                
                metrics = {
                    'test_loss': float(loss),
                    'test_accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            else:
                metrics = {}
            
            logger.info(f"✅ Validation complete: {metrics}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
            return {}
    
    def save_model(self, version_tag: str, metrics: Dict, output_dir: str = "storage/models") -> str:
        """Save trained model"""
        
        if self.model is None:
            logger.error("No model to save")
            return ""
        
        try:
            output_path = Path(output_dir) / self.model_type
            output_path.mkdir(parents=True, exist_ok=True)
            
            model_filename = f"{self.model_type}_{version_tag}"
            model_path = output_path / model_filename
            
            if self.framework == "tensorflow":
                self.model.save(str(model_path))
            elif self.framework == "pytorch":
                torch.save(self.model.state_dict(), str(model_path) + ".pth")
            
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'version_tag': version_tag,
                'framework': self.framework,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'config': self.config
            }
            
            metadata_path = output_path / f"{model_filename}_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
            logger.info(f"💾 Model saved: {model_path}")
            logger.info(f"📋 Metadata saved: {metadata_path}")
            
            return str(model_path)
        
        except Exception as e:
            logger.error(f"❌ Save model error: {e}")
            return ""
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model"""
        try:
            model_path = Path(model_path)
            
            if self.framework == "tensorflow":
                self.model = keras.models.load_model(str(model_path))
            elif self.framework == "pytorch":
                # PyTorch loading requires model architecture
                logger.warning("PyTorch model loading requires architecture definition")
                return False
            
            logger.info(f"✅ Model loaded from {model_path}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Load model error: {e}")
            return False
