"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
LSTM/Transformer Time-Series Layer

Learns next move from sequences, captures:
- Liquidity cycles
- Repeated patterns
- Trend shifts
- Temporal dependencies
- Long-term memory
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class LSTMOutput:
    """Output from LSTM/Transformer layer"""
    direction: str  # 'long', 'short', 'neutral'
    confidence: float  # 0.0 to 1.0
    predicted_move: float  # Predicted price change percentage
    attention_weights: Optional[Dict[str, float]] = None  # Feature attention scores
    sequence_features: Optional[Dict[str, float]] = None  # Sequence-level features


class TimeSeriesLayer:
    """
    LSTM/Transformer-based time series prediction layer.
    
    Uses recurrent/attention mechanisms to learn from historical sequences
    and predict future market movements.
    """
    
    def __init__(self, config):
        """
        Initialize Time Series layer
        
        Args:
            config: LSTMConfig object with hyperparameters
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.sequence_buffer = deque(maxlen=config.sequence_length)
        
        logger.info("Initializing LSTM/Transformer Time-Series Layer")
        self._build_model()
    
    def _build_model(self):
        """Build LSTM or Transformer architecture"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            input_shape = (self.config.sequence_length, len(self.config.features))
            
            inputs = keras.Input(shape=input_shape)
            
            if self.config.use_transformer:
                # Transformer-based architecture
                x = self._build_transformer_block(inputs)
            else:
                # Pure LSTM architecture
                x = inputs
                for i, units in enumerate(self.config.lstm_units):
                    return_sequences = (i < len(self.config.lstm_units) - 1)
                    x = layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        dropout=self.config.dropout_rate,
                        name=f'lstm_{i}'
                    )(x)
            
            # Dense layers
            x = layers.Dense(128, activation='relu', name='dense_1')(x)
            x = layers.Dropout(self.config.dropout_rate, name='dropout_1')(x)
            x = layers.Dense(64, activation='relu', name='dense_2')(x)
            x = layers.Dropout(self.config.dropout_rate, name='dropout_2')(x)
            
            # Output layers
            direction_output = layers.Dense(3, activation='softmax', name='direction')(x)
            move_output = layers.Dense(1, activation='tanh', name='predicted_move')(x)  # -1 to 1
            confidence_output = layers.Dense(1, activation='sigmoid', name='confidence')(x)
            
            self.model = keras.Model(
                inputs=inputs,
                outputs=[direction_output, move_output, confidence_output],
                name='time_series_predictor'
            )
            
            # Compile
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss={
                    'direction': 'categorical_crossentropy',
                    'predicted_move': 'mse',
                    'confidence': 'mse'
                },
                metrics={
                    'direction': 'accuracy',
                    'predicted_move': 'mae',
                    'confidence': 'mae'
                }
            )
            
            logger.info(f"Time-Series model built: {self.model.count_params()} parameters")
            
        except ImportError:
            logger.warning("TensorFlow not available, using lightweight fallback")
            self.model = None
    
    def _build_transformer_block(self, inputs):
        """
        Build transformer encoder block with multi-head attention
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor
        """
        try:
            from tensorflow.keras import layers
            
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=self.config.attention_heads,
                key_dim=64,
                name='multi_head_attention'
            )(inputs, inputs)
            
            attention_output = layers.Dropout(self.config.dropout_rate)(attention_output)
            attention_output = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
            
            # Feed-forward network
            ffn_output = layers.Dense(128, activation='relu')(attention_output)
            ffn_output = layers.Dropout(self.config.dropout_rate)(ffn_output)
            ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
            ffn_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + ffn_output)
            
            # Global average pooling
            output = layers.GlobalAveragePooling1D()(ffn_output)
            
            return output
            
        except Exception as e:
            logger.error(f"Transformer build error: {e}")
            # Fallback to simple LSTM
            return layers.LSTM(128, name='fallback_lstm')(inputs)
    
    def update_sequence(self, market_data: Dict):
        """
        Update the sequence buffer with new market data
        
        Args:
            market_data: Dictionary with current market state
        """
        # Extract features
        feature_vector = []
        
        for feature_name in self.config.features:
            value = market_data.get(feature_name, 0.0)
            feature_vector.append(float(value))
        
        self.sequence_buffer.append(feature_vector)
    
    def predict(self, market_data: Optional[Dict] = None) -> LSTMOutput:
        """
        Make prediction on current sequence
        
        Args:
            market_data: Optional new market data to add to sequence
            
        Returns:
            LSTMOutput with direction, confidence, and predicted move
        """
        try:
            # Update sequence if new data provided
            if market_data:
                self.update_sequence(market_data)
            
            # Need full sequence
            if len(self.sequence_buffer) < self.config.sequence_length:
                logger.debug(f"Insufficient sequence data: {len(self.sequence_buffer)}/{self.config.sequence_length}")
                return LSTMOutput(
                    direction='neutral',
                    confidence=0.0,
                    predicted_move=0.0
                )
            
            # Convert to array
            sequence = np.array(list(self.sequence_buffer), dtype=np.float32)
            
            # Normalize
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.fit(sequence)
            
            normalized = self.scaler.transform(sequence)
            normalized = np.expand_dims(normalized, axis=0)  # Add batch dimension
            
            if self.model and self.is_trained:
                # Neural network prediction
                direction_probs, predicted_move, confidence = self.model.predict(normalized, verbose=0)
                
                direction_idx = np.argmax(direction_probs[0])
                direction_map = {0: 'long', 1: 'short', 2: 'neutral'}
                direction = direction_map[direction_idx]
                
                confidence_val = float(confidence[0][0])
                move_val = float(predicted_move[0][0])
                
                # Get attention weights if transformer
                attention_weights = self._extract_attention_weights(normalized)
                
            else:
                # Fallback: momentum-based prediction
                direction, confidence_val, move_val = self._fallback_momentum_prediction(sequence)
                attention_weights = None
            
            # Compute sequence features
            sequence_features = self._compute_sequence_features(sequence)
            
            return LSTMOutput(
                direction=direction,
                confidence=confidence_val,
                predicted_move=move_val,
                attention_weights=attention_weights,
                sequence_features=sequence_features
            )
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}", exc_info=True)
            return LSTMOutput(
                direction='neutral',
                confidence=0.0,
                predicted_move=0.0
            )
    
    def _fallback_momentum_prediction(self, sequence: np.ndarray) -> Tuple[str, float, float]:
        """
        Fallback momentum-based prediction when neural network unavailable
        
        Args:
            sequence: Sequence array (sequence_length, num_features)
            
        Returns:
            (direction, confidence, predicted_move) tuple
        """
        # Assume first feature is price
        prices = sequence[:, 0]
        
        # Calculate short and long term momentum
        short_window = min(10, len(prices) // 3)
        long_window = min(30, len(prices) // 2)
        
        short_momentum = (prices[-1] - prices[-short_window]) / prices[-short_window] if prices[-short_window] != 0 else 0
        long_momentum = (prices[-1] - prices[-long_window]) / prices[-long_window] if prices[-long_window] != 0 else 0
        
        # Combine momentums
        combined_momentum = 0.7 * short_momentum + 0.3 * long_momentum
        
        # Determine direction
        if combined_momentum > 0.001:  # 0.1% threshold
            direction = 'long'
            predicted_move = combined_momentum
        elif combined_momentum < -0.001:
            direction = 'short'
            predicted_move = combined_momentum
        else:
            direction = 'neutral'
            predicted_move = 0.0
        
        # Calculate confidence based on momentum strength and consistency
        momentum_strength = abs(combined_momentum)
        
        # Check for trend consistency
        recent_moves = np.diff(prices[-10:])
        consistency = np.sum(np.sign(recent_moves) == np.sign(combined_momentum)) / len(recent_moves) if len(recent_moves) > 0 else 0
        
        confidence = min(0.5 + momentum_strength * 50 + consistency * 0.3, 0.95)
        
        return direction, confidence, predicted_move
    
    def _extract_attention_weights(self, sequence: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Extract attention weights from transformer model
        
        Args:
            sequence: Input sequence
            
        Returns:
            Dictionary of feature attention weights
        """
        if not self.config.use_transformer or not self.model:
            return None
        
        try:
            # Get attention layer
            attention_layer = None
            for layer in self.model.layers:
                if 'attention' in layer.name.lower():
                    attention_layer = layer
                    break
            
            if attention_layer is None:
                return None
            
            # Extract weights (simplified - would need proper implementation)
            weights = {}
            for i, feature_name in enumerate(self.config.features):
                # Placeholder - real implementation would extract actual attention scores
                weights[feature_name] = float(1.0 / len(self.config.features))
            
            return weights
            
        except Exception as e:
            logger.debug(f"Could not extract attention weights: {e}")
            return None
    
    def _compute_sequence_features(self, sequence: np.ndarray) -> Dict[str, float]:
        """
        Compute aggregate features from the sequence
        
        Args:
            sequence: Sequence array
            
        Returns:
            Dictionary of sequence-level features
        """
        features = {}
        
        try:
            # Trend strength
            prices = sequence[:, 0]
            trend = np.polyfit(range(len(prices)), prices, 1)[0]
            features['trend_strength'] = float(trend)
            
            # Volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            features['volatility'] = float(volatility)
            
            # Momentum oscillation
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-30:]) if len(prices) >= 30 else np.mean(prices)
            features['momentum_oscillation'] = float((short_ma - long_ma) / long_ma) if long_ma != 0 else 0.0
            
            # Volume trend (if volume is in features)
            if len(sequence[0]) > 1:
                volumes = sequence[:, 1]
                volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
                features['volume_trend'] = float(volume_trend)
            
            # Price acceleration
            if len(prices) >= 3:
                acceleration = prices[-1] - 2*prices[-2] + prices[-3]
                features['price_acceleration'] = float(acceleration)
            
        except Exception as e:
            logger.debug(f"Error computing sequence features: {e}")
        
        return features
    
    def train(self, training_data: List[Tuple[List[Dict], str, float, float]], epochs: int = None):
        """
        Train the LSTM/Transformer model
        
        Args:
            training_data: List of (sequence_data, direction, confidence, actual_move) tuples
            epochs: Number of training epochs (overrides config)
        """
        if not self.model:
            logger.warning("No TensorFlow model available for training")
            return
        
        logger.info(f"Training Time-Series model with {len(training_data)} sequences")
        
        # Prepare training data
        X = []
        y_direction = []
        y_move = []
        y_confidence = []
        
        direction_map = {'long': 0, 'short': 1, 'neutral': 2}
        
        for sequence_data, direction, confidence, actual_move in training_data:
            # Build sequence
            sequence = []
            for data_point in sequence_data:
                feature_vector = [data_point.get(f, 0.0) for f in self.config.features]
                sequence.append(feature_vector)
            
            # Pad if needed
            while len(sequence) < self.config.sequence_length:
                sequence.insert(0, sequence[0])
            
            sequence = sequence[-self.config.sequence_length:]
            
            X.append(sequence)
            
            # One-hot encode direction
            direction_encoded = [0, 0, 0]
            direction_encoded[direction_map.get(direction, 2)] = 1
            y_direction.append(direction_encoded)
            
            y_move.append([actual_move])
            y_confidence.append([confidence])
        
        X = np.array(X, dtype=np.float32)
        y_direction = np.array(y_direction)
        y_move = np.array(y_move)
        y_confidence = np.array(y_confidence)
        
        # Normalize X
        for i in range(X.shape[0]):
            if self.scaler is None:
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.scaler.fit(X[i])
            X[i] = self.scaler.transform(X[i])
        
        # Train model
        epochs = epochs or self.config.epochs_per_cycle
        
        self.model.fit(
            X,
            {
                'direction': y_direction,
                'predicted_move': y_move,
                'confidence': y_confidence
            },
            batch_size=self.config.batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        self.is_trained = True
        logger.info("Time-Series model training completed")
    
    def save(self, filepath: str):
        """Save model weights"""
        if self.model:
            self.model.save_weights(filepath)
            logger.info(f"Time-Series model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model weights"""
        if self.model:
            self.model.load_weights(filepath)
            self.is_trained = True
            logger.info(f"Time-Series model loaded from {filepath}")
    
    def reset_sequence(self):
        """Clear the sequence buffer"""
        self.sequence_buffer.clear()
        logger.debug("Sequence buffer reset")
