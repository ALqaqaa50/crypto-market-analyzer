import numpy as np
import logging
from typing import Dict, List
from collections import deque
from scipy import stats
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeDNAClassifier:
    def __init__(self, lookback_window: int = 100, volatility_window: int = 20):
        self.lookback_window = lookback_window
        self.volatility_window = volatility_window
        
        self.price_buffer = deque(maxlen=lookback_window)
        self.volume_buffer = deque(maxlen=lookback_window)
        self.orderflow_buffer = deque(maxlen=lookback_window)
        
        self.current_regime = "unknown"
        self.regime_confidence = 0.0
        self.regime_history = deque(maxlen=1000)
        
        self.regime_features = {}
        
        logger.info(f"RegimeDNAClassifier initialized: lookback={lookback_window}")
    
    def update(self, price: float, volume: float, buy_volume: float, sell_volume: float,
               bid: float, ask: float):
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        orderflow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-8)
        self.orderflow_buffer.append(orderflow_imbalance)
    
    def classify(self) -> Dict:
        if len(self.price_buffer) < self.lookback_window:
            return {
                'regime': 'warming_up',
                'confidence': 0.0,
                'features': {}
            }
        
        self.regime_features = self._extract_features()
        
        regime_scores = {
            'trend_up': self._score_trend_up(),
            'trend_down': self._score_trend_down(),
            'range': self._score_range(),
            'low_volatility': self._score_low_volatility(),
            'high_volatility': self._score_high_volatility(),
            'choppy': self._score_choppy(),
            'parabolic_run': self._score_parabolic(),
            'flash_crash': self._score_flash_crash(),
            'whale_accumulation': self._score_whale_accumulation(),
            'whale_distribution': self._score_whale_distribution(),
            'spoofing_zone': self._score_spoofing(),
            'stop_hunt_zone': self._score_stop_hunt()
        }
        
        self.current_regime = max(regime_scores, key=regime_scores.get)
        self.regime_confidence = regime_scores[self.current_regime]
        
        self.regime_history.append({
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'timestamp': len(self.regime_history)
        })
        
        return {
            'regime': self.current_regime,
            'confidence': self.regime_confidence,
            'scores': regime_scores,
            'features': self.regime_features
        }
    
    def _extract_features(self) -> Dict:
        prices = np.array(list(self.price_buffer))
        volumes = np.array(list(self.volume_buffer))
        orderflows = np.array(list(self.orderflow_buffer))
        
        returns = np.diff(prices) / prices[:-1]
        
        volatility_short = np.std(returns[-self.volatility_window:])
        volatility_long = np.std(returns)
        
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
        
        price_trend = (prices[-1] - prices[0]) / prices[0]
        
        volume_trend = (volumes[-20:].mean() - volumes[-50:].mean()) / (volumes[-50:].mean() + 1e-8) if len(volumes) >= 50 else 0
        
        orderflow_trend = np.mean(orderflows[-20:])
        
        slope, _, r_value, _, _ = stats.linregress(np.arange(len(prices)), prices)
        
        highs = [prices[i] for i in range(1, len(prices)-1) if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
        lows = [prices[i] for i in range(1, len(prices)-1) if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        
        swing_ratio = len(highs) / (len(lows) + 1e-8)
        
        atr = np.mean(np.abs(returns[-14:])) if len(returns) >= 14 else volatility_short
        
        return {
            'price_trend': price_trend,
            'volatility_short': volatility_short,
            'volatility_long': volatility_long,
            'volatility_ratio': volatility_short / (volatility_long + 1e-8),
            'sma_20': sma_20,
            'sma_50': sma_50,
            'price_to_sma20': (prices[-1] - sma_20) / sma_20,
            'price_to_sma50': (prices[-1] - sma_50) / sma_50,
            'volume_trend': volume_trend,
            'orderflow_trend': orderflow_trend,
            'trend_strength': abs(r_value),
            'trend_slope': slope,
            'swing_ratio': swing_ratio,
            'atr': atr,
            'recent_return': returns[-1] if len(returns) > 0 else 0,
            'recent_return_5': np.mean(returns[-5:]) if len(returns) >= 5 else 0
        }
    
    def _score_trend_up(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['price_trend'] > 0.02:
            score += 0.3
        if f['trend_slope'] > 0 and f['trend_strength'] > 0.8:
            score += 0.3
        if f['price_to_sma20'] > 0 and f['price_to_sma50'] > 0:
            score += 0.2
        if f['orderflow_trend'] > 0.1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_trend_down(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['price_trend'] < -0.02:
            score += 0.3
        if f['trend_slope'] < 0 and f['trend_strength'] > 0.8:
            score += 0.3
        if f['price_to_sma20'] < 0 and f['price_to_sma50'] < 0:
            score += 0.2
        if f['orderflow_trend'] < -0.1:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_range(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if abs(f['price_trend']) < 0.01:
            score += 0.3
        if f['trend_strength'] < 0.5:
            score += 0.3
        if 0.8 < f['swing_ratio'] < 1.2:
            score += 0.2
        if f['volatility_ratio'] < 1.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_low_volatility(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['volatility_short'] < f['volatility_long'] * 0.5:
            score += 0.5
        if f['atr'] < np.percentile([f['atr']], 25):
            score += 0.3
        if abs(f['price_trend']) < 0.005:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_high_volatility(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['volatility_short'] > f['volatility_long'] * 1.5:
            score += 0.5
        if f['atr'] > np.percentile([f['atr']], 75):
            score += 0.3
        if abs(f['recent_return']) > 0.01:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_choppy(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['volatility_ratio'] > 1.5:
            score += 0.3
        if f['trend_strength'] < 0.3:
            score += 0.3
        if f['swing_ratio'] > 1.5 or f['swing_ratio'] < 0.67:
            score += 0.2
        if abs(f['orderflow_trend']) < 0.05:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_parabolic(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if abs(f['price_trend']) > 0.05:
            score += 0.3
        if f['volatility_short'] > f['volatility_long'] * 1.3:
            score += 0.3
        if abs(f['trend_slope']) > np.percentile([abs(f['trend_slope'])], 90):
            score += 0.2
        if f['volume_trend'] > 0.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _score_flash_crash(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['recent_return'] < -0.02:
            score += 0.4
        if f['volatility_short'] > f['volatility_long'] * 2.0:
            score += 0.3
        if f['volume_trend'] > 1.0:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_whale_accumulation(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['orderflow_trend'] > 0.2:
            score += 0.4
        if f['volume_trend'] > 0.3:
            score += 0.3
        if abs(f['price_trend']) < 0.02:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_whale_distribution(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if f['orderflow_trend'] < -0.2:
            score += 0.4
        if f['volume_trend'] > 0.3:
            score += 0.3
        if abs(f['price_trend']) < 0.02:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_spoofing(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if abs(f['orderflow_trend']) > 0.3:
            score += 0.4
        if f['volatility_short'] > f['volatility_long'] * 1.2:
            score += 0.3
        if abs(f['recent_return']) > 0.005:
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_stop_hunt(self) -> float:
        f = self.regime_features
        score = 0.0
        
        if abs(f['recent_return']) > 0.01:
            score += 0.3
        if f['recent_return_5'] * f['recent_return'] < 0:
            score += 0.3
        if f['volume_trend'] > 0.5:
            score += 0.2
        if abs(f['orderflow_trend']) > 0.2:
            score += 0.2
        
        return min(score, 1.0)
    
    def get_regime_statistics(self) -> Dict:
        if len(self.regime_history) == 0:
            return {}
        
        regime_counts = {}
        for entry in self.regime_history:
            regime = entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(self.regime_history)
        regime_percentages = {k: v / total * 100 for k, v in regime_counts.items()}
        
        return {
            'current_regime': self.current_regime,
            'current_confidence': self.regime_confidence,
            'regime_counts': regime_counts,
            'regime_percentages': regime_percentages,
            'history_length': total
        }
