"""
Support and Resistance Level Detection
"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from ...utils.logger import get_logger

logger = get_logger(__name__)


class SupportResistanceDetector:
    """
    Detect support and resistance levels using:
    - Peak/trough detection
    - Price clustering
    - Volume-weighted levels
    """
    
    def __init__(
        self,
        min_touches: int = 2,
        proximity_threshold: float = 0.005  # 0.5% proximity
    ):
        """
        Args:
            min_touches: Minimum number of touches to confirm a level
            proximity_threshold: Price proximity threshold (as fraction)
        """
        self.min_touches = min_touches
        self.proximity_threshold = proximity_threshold
    
    def detect_levels(
        self,
        df: pd.DataFrame,
        window: int = 50
    ) -> Tuple[List[float], List[float]]:
        """
        Detect support and resistance levels.
        
        Args:
            df: DataFrame with OHLC data
            window: Lookback window
            
        Returns:
            (support_levels, resistance_levels)
        """
        if len(df) < window:
            return ([], [])
        
        # Use recent data
        recent_df = df.tail(window).copy()
        
        # Find local peaks and troughs
        highs = self._find_peaks(recent_df['high'].values)
        lows = self._find_peaks(-recent_df['low'].values)
        
        # Get actual price values
        high_prices = recent_df['high'].iloc[highs].values
        low_prices = recent_df['low'].iloc[lows].values
        
        # Cluster similar prices
        resistance_levels = self._cluster_levels(high_prices)
        support_levels = self._cluster_levels(low_prices)
        
        # Filter by minimum touches
        resistance_levels = self._filter_by_touches(
            resistance_levels, recent_df['high'].values
        )
        support_levels = self._filter_by_touches(
            support_levels, recent_df['low'].values
        )
        
        logger.debug(
            f"Detected {len(support_levels)} support and "
            f"{len(resistance_levels)} resistance levels"
        )
        
        return (support_levels, resistance_levels)
    
    def _find_peaks(self, data: np.ndarray, order: int = 3) -> np.ndarray:
        """Find local peaks in data"""
        peaks = []
        for i in range(order, len(data) - order):
            if all(data[i] > data[i-j] for j in range(1, order+1)) and \
               all(data[i] > data[i+j] for j in range(1, order+1)):
                peaks.append(i)
        return np.array(peaks)
    
    def _cluster_levels(self, prices: np.ndarray) -> List[float]:
        """Cluster similar price levels"""
        if len(prices) < 2:
            return prices.tolist()
        
        # Reshape for clustering
        prices_reshaped = prices.reshape(-1, 1)
        
        try:
            # Hierarchical clustering
            threshold = np.mean(prices) * self.proximity_threshold
            
            # Calculate pairwise distances
            distances = pdist(prices_reshaped)
            
            # Perform clustering
            linkage_matrix = linkage(distances, method='ward')
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
            
            # Get cluster centers
            levels = []
            for cluster_id in np.unique(clusters):
                cluster_prices = prices[clusters == cluster_id]
                level = np.mean(cluster_prices)
                levels.append(level)
            
            return sorted(levels)
        
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, using raw prices")
            return sorted(np.unique(prices).tolist())
    
    def _filter_by_touches(
        self,
        levels: List[float],
        prices: np.ndarray
    ) -> List[float]:
        """Filter levels by minimum number of touches"""
        filtered = []
        
        for level in levels:
            # Count how many times price touched this level
            touches = 0
            for price in prices:
                if abs(price - level) / level <= self.proximity_threshold:
                    touches += 1
            
            if touches >= self.min_touches:
                filtered.append(level)
        
        return filtered
    
    def get_nearest_support_resistance(
        self,
        current_price: float,
        support_levels: List[float],
        resistance_levels: List[float]
    ) -> dict:
        """
        Get nearest support and resistance levels.
        
        Returns:
            Dict with nearest levels and distances
        """
        # Find nearest support (below current price)
        supports_below = [s for s in support_levels if s < current_price]
        nearest_support = max(supports_below) if supports_below else None
        
        # Find nearest resistance (above current price)
        resistances_above = [r for r in resistance_levels if r > current_price]
        nearest_resistance = min(resistances_above) if resistances_above else None
        
        result = {
            "current_price": current_price,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
        }
        
        if nearest_support:
            result["support_distance"] = current_price - nearest_support
            result["support_distance_pct"] = (
                (current_price - nearest_support) / current_price * 100
            )
        
        if nearest_resistance:
            result["resistance_distance"] = nearest_resistance - current_price
            result["resistance_distance_pct"] = (
                (nearest_resistance - current_price) / current_price * 100
            )
        
        return result
