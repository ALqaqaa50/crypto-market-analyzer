"""
PHASE 4: Dataset Builder
Builds training datasets from logged experience data
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    Build training datasets from logged experiences
    Supports windowed time-series for CNN/LSTM training
    """
    
    def __init__(
        self,
        window_size: int = 50,
        prediction_horizon: int = 10,
        target_type: str = "direction"  # direction, return, outcome
    ):
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.target_type = target_type
        
        logger.info(f"📊 DatasetBuilder initialized (window={window_size}, horizon={prediction_horizon}, target={target_type})")
    
    def build_from_logs(
        self,
        data: pd.DataFrame,
        features: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build X, y datasets from log dataframe"""
        
        if data.empty:
            logger.warning("Empty dataframe provided")
            return np.array([]), np.array([])
        
        # Default features
        if features is None:
            features = [
                'price', 'bid', 'ask', 'spread',
                'buy_volume', 'sell_volume', 'orderbook_imbalance',
                'confidence'
            ]
        
        # Filter available features
        available_features = [f for f in features if f in data.columns]
        
        if not available_features:
            logger.error(f"No features found in data. Columns: {data.columns.tolist()}")
            return np.array([]), np.array([])
        
        logger.info(f"Using features: {available_features}")
        
        # Extract feature matrix
        feature_data = data[available_features].values
        
        # Build windowed sequences
        X, y = [], []
        
        for i in range(len(feature_data) - self.window_size - self.prediction_horizon):
            # Input window
            x_window = feature_data[i:i + self.window_size]
            
            # Target (future price direction/return)
            if self.target_type == "direction":
                target = self._compute_direction_target(data, i)
            elif self.target_type == "return":
                target = self._compute_return_target(data, i)
            elif self.target_type == "outcome":
                target = self._compute_outcome_target(data, i)
            else:
                target = 0
            
            if target is not None:
                X.append(x_window)
                y.append(target)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Built dataset: X.shape={X.shape}, y.shape={y.shape}")
        
        return X, y
    
    def _compute_direction_target(self, data: pd.DataFrame, index: int) -> Optional[int]:
        """Compute future price direction (0=down, 1=up)"""
        try:
            current_price = data.iloc[index]['price']
            future_price = data.iloc[index + self.window_size + self.prediction_horizon]['price']
            
            return 1 if future_price > current_price else 0
        except:
            return None
    
    def _compute_return_target(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """Compute future return"""
        try:
            current_price = data.iloc[index]['price']
            future_price = data.iloc[index + self.window_size + self.prediction_horizon]['price']
            
            return (future_price - current_price) / current_price
        except:
            return None
    
    def _compute_outcome_target(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """Compute trade outcome if available"""
        try:
            row = data.iloc[index + self.window_size + self.prediction_horizon]
            
            if 'pnl' in row and not pd.isna(row['pnl']):
                return 1 if row['pnl'] > 0 else 0
            
            return self._compute_direction_target(data, index)
        except:
            return None
    
    def normalize_features(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize features (zero mean, unit variance)"""
        if X.size == 0:
            return X, {}
        
        mean = X.mean(axis=(0, 1))
        std = X.std(axis=(0, 1))
        std[std == 0] = 1  # Avoid division by zero
        
        X_norm = (X - mean) / std
        
        normalization_params = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        
        logger.info(f"✅ Features normalized (mean={mean.mean():.4f}, std={std.mean():.4f})")
        
        return X_norm, normalization_params
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split into train/test sets"""
        if X.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        split_index = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        logger.info(f"✅ Train/Test split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_rl_dataset(
        self,
        data: pd.DataFrame
    ) -> List[Dict]:
        """Build dataset for RL training (state-action-reward-next_state)"""
        
        rl_data = []
        
        for i in range(len(data) - 1):
            if data.iloc[i].get('type') == 'trade' and i + 1 < len(data):
                
                state = {
                    'price': data.iloc[i].get('price', 0),
                    'confidence': data.iloc[i].get('confidence', 0),
                    'regime': data.iloc[i].get('regime', 'unknown')
                }
                
                action = 1 if data.iloc[i].get('action') == 'BUY' else 0
                
                # Find outcome
                reward = 0
                for j in range(i + 1, min(i + 100, len(data))):
                    if data.iloc[j].get('type') == 'outcome':
                        reward = data.iloc[j].get('pnl', 0)
                        break
                
                next_state = {
                    'price': data.iloc[i + 1].get('price', 0),
                    'confidence': data.iloc[i + 1].get('confidence', 0),
                    'regime': data.iloc[i + 1].get('regime', 'unknown')
                }
                
                rl_data.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state
                })
        
        logger.info(f"✅ Built RL dataset: {len(rl_data)} transitions")
        
        return rl_data
    
    def get_stats(self, data: pd.DataFrame) -> Dict:
        """Get dataset statistics"""
        if data.empty:
            return {}
        
        stats = {
            'total_records': len(data),
            'date_range': {
                'start': data.iloc[0].get('timestamp', 'unknown'),
                'end': data.iloc[-1].get('timestamp', 'unknown')
            },
            'record_types': data['type'].value_counts().to_dict() if 'type' in data.columns else {},
            'price_range': {
                'min': data['price'].min() if 'price' in data.columns else 0,
                'max': data['price'].max() if 'price' in data.columns else 0
            }
        }
        
        return stats
