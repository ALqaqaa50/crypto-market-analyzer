"""
Volume Profile Analyzer
Identifies high-volume price levels (value areas)
"""
from typing import Dict, List, Tuple
from collections import defaultdict

from ...utils.logger import get_logger


logger = get_logger(__name__)


class VolumeProfileAnalyzer:
    """
    Volume Profile analysis.
    
    Identifies:
    - Point of Control (POC): Price level with highest volume
    - Value Area: Price range containing 70% of volume
    - High Volume Nodes (HVN): Support/resistance levels
    - Low Volume Nodes (LVN): Potential breakout zones
    """
    
    def __init__(self, price_bucket_size: float = 100.0):
        """
        Args:
            price_bucket_size: Size of price buckets (e.g., $100)
        """
        self.bucket_size = price_bucket_size
        self.volume_by_price: Dict[float, float] = defaultdict(float)
        self.total_volume = 0.0
    
    def add_trade(self, price: float, volume: float):
        """Add a trade to the volume profile"""
        # Round price to nearest bucket
        bucket = round(price / self.bucket_size) * self.bucket_size
        self.volume_by_price[bucket] += volume
        self.total_volume += volume
    
    def get_point_of_control(self) -> Tuple[float, float]:
        """
        Get Point of Control (price with highest volume).
        
        Returns:
            (poc_price, poc_volume)
        """
        if not self.volume_by_price:
            return (0.0, 0.0)
        
        poc_price = max(self.volume_by_price, key=self.volume_by_price.get)
        poc_volume = self.volume_by_price[poc_price]
        
        return (poc_price, poc_volume)
    
    def get_value_area(self, percentage: float = 0.70) -> Tuple[float, float]:
        """
        Get value area containing X% of volume.
        
        Args:
            percentage: Percentage of volume to include (default 70%)
            
        Returns:
            (value_area_low, value_area_high)
        """
        if not self.volume_by_price:
            return (0.0, 0.0)
        
        # Sort prices by volume (descending)
        sorted_prices = sorted(
            self.volume_by_price.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        target_volume = self.total_volume * percentage
        cumulative_volume = 0.0
        value_prices = []
        
        for price, volume in sorted_prices:
            value_prices.append(price)
            cumulative_volume += volume
            if cumulative_volume >= target_volume:
                break
        
        if not value_prices:
            return (0.0, 0.0)
        
        value_area_low = min(value_prices)
        value_area_high = max(value_prices)
        
        return (value_area_low, value_area_high)
    
    def get_high_volume_nodes(self, threshold_percentile: float = 0.80) -> List[float]:
        """
        Get High Volume Nodes (HVNs).
        
        These are price levels with volume above threshold percentile.
        Often act as support/resistance.
        
        Args:
            threshold_percentile: Percentile threshold (0-1)
            
        Returns:
            List of HVN prices
        """
        if not self.volume_by_price:
            return []
        
        volumes = list(self.volume_by_price.values())
        volumes.sort()
        threshold_idx = int(len(volumes) * threshold_percentile)
        threshold_volume = volumes[threshold_idx] if threshold_idx < len(volumes) else 0
        
        hvns = [
            price for price, volume in self.volume_by_price.items()
            if volume >= threshold_volume
        ]
        
        return sorted(hvns)
    
    def get_low_volume_nodes(self, threshold_percentile: float = 0.20) -> List[float]:
        """
        Get Low Volume Nodes (LVNs).
        
        These are price levels with low volume.
        Often represent breakout zones or gaps.
        
        Args:
            threshold_percentile: Percentile threshold (0-1)
            
        Returns:
            List of LVN prices
        """
        if not self.volume_by_price:
            return []
        
        volumes = list(self.volume_by_price.values())
        volumes.sort()
        threshold_idx = int(len(volumes) * threshold_percentile)
        threshold_volume = volumes[threshold_idx] if threshold_idx < len(volumes) else float('inf')
        
        lvns = [
            price for price, volume in self.volume_by_price.items()
            if volume <= threshold_volume
        ]
        
        return sorted(lvns)
    
    def get_profile_summary(self) -> Dict:
        """Get complete volume profile summary"""
        poc_price, poc_volume = self.get_point_of_control()
        val, vah = self.get_value_area()
        hvns = self.get_high_volume_nodes()
        lvns = self.get_low_volume_nodes()
        
        return {
            'point_of_control': {
                'price': poc_price,
                'volume': poc_volume,
            },
            'value_area': {
                'low': val,
                'high': vah,
                'range': vah - val,
            },
            'high_volume_nodes': hvns,
            'low_volume_nodes': lvns,
            'total_volume': self.total_volume,
            'price_levels': len(self.volume_by_price),
        }
    
    def reset(self):
        """Reset volume profile"""
        self.volume_by_price.clear()
        self.total_volume = 0.0