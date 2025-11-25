"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
CNN Layer - Micro-Pattern Detection

Convolutional Neural Network for learning:
- Micro-patterns from recent candles
- Price microstructure
- Trend momentum
- Wick behavior
- Volume imbalance
- Volatility bursts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CNNOutput:
    """Output from CNN layer"""
    direction: str  # 'long', 'short', 'neutral'
    confidence: float  # 0.0 to 1.0
    features: Dict[str, float]  # Detected features
    pattern_type: Optional[str] = None  # e.g., 'bullish_engulfing', 'head_shoulders'


class CNNLayer:
    """
    CNN-based micro-pattern detector for price action analysis.
    
    Uses convolutional layers to detect local patterns in candlestick data,
    similar to image pattern recognition but applied to financial time series.
    """
    
    def __init__(self, config):
        """
        Initialize CNN layer
        
        Args:
            config: CNNConfig object with hyperparameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        logger.info("Initializing CNN Layer for micro-pattern detection")
        self._build_model()
    
    def _build_model(self):
        """Build the CNN architecture"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            # Input shape: (sequence_length, num_features)
            input_shape = (self.config.input_sequence_length, len(self.config.features_to_extract))
            
            inputs = keras.Input(shape=input_shape)
            
            # Convolutional blocks
            x = inputs
            for i, (filters, kernel_size, pool_size) in enumerate(zip(
                self.config.conv_filters,
                self.config.kernel_sizes,
                self.config.pool_sizes
            )):
                x = layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    padding='same',
                    name=f'conv1d_{i}'
                )(x)
                x = layers.BatchNormalization(name=f'bn_{i}')(x)
                x = layers.MaxPooling1D(pool_size=pool_size, name=f'pool_{i}')(x)
                x = layers.Dropout(self.config.dropout_rate, name=f'dropout_{i}')(x)
            
            # Flatten and dense layers
            x = layers.Flatten(name='flatten')(x)
            
            for i, units in enumerate(self.config.dense_layers):
                x = layers.Dense(units, activation='relu', name=f'dense_{i}')(x)
                x = layers.Dropout(self.config.dropout_rate, name=f'dense_dropout_{i}')(x)
            
            # Output layers
            direction_output = layers.Dense(3, activation='softmax', name='direction')(x)  # long/short/neutral
            confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
            
            self.model = keras.Model(
                inputs=inputs,
                outputs=[direction_output, confidence_output],
                name='cnn_pattern_detector'
            )
            
            # Compile model
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss={
                    'direction': 'categorical_crossentropy',
                    'confidence': 'mse'
                },
                metrics={
                    'direction': 'accuracy',
                    'confidence': 'mae'
                }
            )
            
            logger.info(f"CNN model built successfully: {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available, using lightweight fallback")
            self.model = None
    
    def extract_features(self, candles: List[Dict]) -> np.ndarray:
        """
        Extract features from candlestick data
        
        Args:
            candles: List of candle dictionaries with OHLCV data
            
        Returns:
            Feature matrix (sequence_length, num_features)
        """
        if len(candles) < self.config.input_sequence_length:
            # Pad with zeros if not enough data
            padding = self.config.input_sequence_length - len(candles)
            candles = [candles[0]] * padding + candles
        
        # Take last N candles
        candles = candles[-self.config.input_sequence_length:]
        
        features = []
        for candle in candles:
            feature_vector = []
            
            # Basic OHLCV
            if 'open' in self.config.features_to_extract:
                feature_vector.append(float(candle.get('open', 0)))
            if 'high' in self.config.features_to_extract:
                feature_vector.append(float(candle.get('high', 0)))
            if 'low' in self.config.features_to_extract:
                feature_vector.append(float(candle.get('low', 0)))
            if 'close' in self.config.features_to_extract:
                feature_vector.append(float(candle.get('close', 0)))
            if 'volume' in self.config.features_to_extract:
                feature_vector.append(float(candle.get('volume', 0)))
            
            # Derived features
            open_price = float(candle.get('open', 0))
            high_price = float(candle.get('high', 0))
            low_price = float(candle.get('low', 0))
            close_price = float(candle.get('close', 0))
            
            if 'wick_ratio' in self.config.features_to_extract:
                body = abs(close_price - open_price)
                total_range = high_price - low_price
                wick_ratio = 1.0 - (body / total_range) if total_range > 0 else 0
                feature_vector.append(wick_ratio)
            
            if 'body_ratio' in self.config.features_to_extract:
                body = abs(close_price - open_price)
                total_range = high_price - low_price
                body_ratio = body / total_range if total_range > 0 else 0
                feature_vector.append(body_ratio)
            
            if 'spread' in self.config.features_to_extract:
                spread = (high_price - low_price) / close_price if close_price > 0 else 0
                feature_vector.append(spread)
            
            features.append(feature_vector)
        
        feature_matrix = np.array(features, dtype=np.float32)
        
        # Normalize features
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(feature_matrix)
        
        normalized = self.scaler.transform(feature_matrix)
        return normalized
    
    def predict(self, candles: List[Dict]) -> CNNOutput:
        """
        Make prediction on current market state
        
        Args:
            candles: List of recent candle dictionaries
            
        Returns:
            CNNOutput with direction, confidence, and features
        """
        try:
            # Extract features
            features = self.extract_features(candles)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            
            if self.model and self.is_trained:
                # Neural network prediction
                direction_probs, confidence = self.model.predict(features, verbose=0)
                
                direction_idx = np.argmax(direction_probs[0])
                direction_map = {0: 'long', 1: 'short', 2: 'neutral'}
                direction = direction_map[direction_idx]
                
                confidence_val = float(confidence[0][0])
                
            else:
                # Fallback: rule-based pattern detection
                direction, confidence_val = self._fallback_pattern_detection(candles)
            
            # Detect specific patterns
            pattern_type = self._detect_pattern_type(candles)
            
            # Extract feature importance
            feature_dict = self._compute_feature_importance(features[0])
            
            return CNNOutput(
                direction=direction,
                confidence=confidence_val,
                features=feature_dict,
                pattern_type=pattern_type
            )
            
        except Exception as e:
            logger.error(f"CNN prediction error: {e}", exc_info=True)
            return CNNOutput(
                direction='neutral',
                confidence=0.0,
                features={},
                pattern_type=None
            )
    
    def _fallback_pattern_detection(self, candles: List[Dict]) -> Tuple[str, float]:
        """
        Rule-based pattern detection when neural network unavailable
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            (direction, confidence) tuple
        """
        if len(candles) < 3:
            return 'neutral', 0.0
        
        last_3 = candles[-3:]
        
        # Simple trend detection
        closes = [float(c.get('close', 0)) for c in last_3]
        
        if closes[-1] > closes[-2] > closes[-3]:
            # Uptrend
            momentum = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] > 0 else 0
            confidence = min(0.5 + abs(momentum) * 10, 0.95)
            return 'long', confidence
        
        elif closes[-1] < closes[-2] < closes[-3]:
            # Downtrend
            momentum = (closes[-3] - closes[-1]) / closes[-3] if closes[-3] > 0 else 0
            confidence = min(0.5 + abs(momentum) * 10, 0.95)
            return 'short', confidence
        
        else:
            return 'neutral', 0.3
    
    def _detect_pattern_type(self, candles: List[Dict]) -> Optional[str]:
        """
        Detect specific candlestick patterns
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Pattern name or None
        """
        if len(candles) < 2:
            return None
        
        last_2 = candles[-2:]
        
        prev_candle = last_2[0]
        curr_candle = last_2[1]
        
        prev_open = float(prev_candle.get('open', 0))
        prev_close = float(prev_candle.get('close', 0))
        prev_high = float(prev_candle.get('high', 0))
        prev_low = float(prev_candle.get('low', 0))
        
        curr_open = float(curr_candle.get('open', 0))
        curr_close = float(curr_candle.get('close', 0))
        curr_high = float(curr_candle.get('high', 0))
        curr_low = float(curr_candle.get('low', 0))
        
        # Bullish engulfing
        if (prev_close < prev_open and  # Previous bearish
            curr_close > curr_open and  # Current bullish
            curr_open < prev_close and  # Opens below prev close
            curr_close > prev_open):    # Closes above prev open
            return 'bullish_engulfing'
        
        # Bearish engulfing
        if (prev_close > prev_open and  # Previous bullish
            curr_close < curr_open and  # Current bearish
            curr_open > prev_close and  # Opens above prev close
            curr_close < prev_open):    # Closes below prev open
            return 'bearish_engulfing'
        
        # Doji
        body = abs(curr_close - curr_open)
        total_range = curr_high - curr_low
        if total_range > 0 and body / total_range < 0.1:
            return 'doji'
        
        # Hammer
        if curr_close > curr_open:
            lower_wick = curr_open - curr_low
            upper_wick = curr_high - curr_close
            body = curr_close - curr_open
            if lower_wick > 2 * body and upper_wick < body * 0.3:
                return 'hammer'
        
        return None
    
    def _compute_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """
        Compute feature importance scores
        
        Args:
            features: Feature array
            
        Returns:
            Dictionary of feature importance scores
        """
        importance = {}
        
        # Simple statistical importance based on magnitude
        for i, feature_name in enumerate(self.config.features_to_extract):
            if i < len(features):
                # Average across sequence
                importance[feature_name] = float(np.mean(np.abs(features[:, i])))
        
        return importance
    
    def train(self, training_data: List[Tuple[List[Dict], str, float]], epochs: int = None):
        """
        Train the CNN model
        
        Args:
            training_data: List of (candles, direction, confidence) tuples
            epochs: Number of training epochs (overrides config)
        """
        if not self.model:
            logger.warning("No TensorFlow model available for training")
            return
        
        logger.info(f"Training CNN with {len(training_data)} samples")
        
        # Prepare training data
        X = []
        y_direction = []
        y_confidence = []
        
        direction_map = {'long': 0, 'short': 1, 'neutral': 2}
        
        for candles, direction, confidence in training_data:
            features = self.extract_features(candles)
            X.append(features)
            
            # One-hot encode direction
            direction_encoded = [0, 0, 0]
            direction_encoded[direction_map.get(direction, 2)] = 1
            y_direction.append(direction_encoded)
            
            y_confidence.append([confidence])
        
        X = np.array(X)
        y_direction = np.array(y_direction)
        y_confidence = np.array(y_confidence)
        
        # Train model
        epochs = epochs or self.config.epochs_per_cycle
        
        self.model.fit(
            X,
            {'direction': y_direction, 'confidence': y_confidence},
            batch_size=self.config.batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("CNN training completed")
    
    def save(self, filepath: str):
        """Save model weights"""
        if self.model:
            self.model.save_weights(filepath)
            logger.info(f"CNN model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        if self.model:
            self.model.load_weights(filepath)
            self.is_trained = True
            logger.info(f"CNN model loaded from {filepath}")
