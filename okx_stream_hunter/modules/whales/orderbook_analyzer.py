"""
Order Book Whale Analyzer
Detects whale movements in order book
"""
from typing import Dict, List, Optional, Tuple

from ...utils.logger import get_logger


logger = get_logger(__name__)


class OrderBookWhaleAnalyzer:
    """
    Analyze order book for whale activity.
    
    Detects:
    - Large walls (bid/ask)
    - Sudden appearance/disappearance of large orders
    - Order book imbalance
    """
    
    def __init__(self, whale_order_threshold: float = 10.0):
        """
        Args:
            whale_order_threshold: Minimum BTC size to consider as whale order
        """
        self.whale_threshold = whale_order_threshold
        self.previous_bids: List[Tuple[float, float]] = []
        self.previous_asks: List[Tuple[float, float]] = []
    
    def analyze_orderbook(
        self,
        bids: List[List],  # [[price, size], ...]
        asks: List[List],
        price: float
    ) -> Dict:
        """
        Analyze order book for whale activity.
        
        Args:
            bids: List of [price, size] for bids
            asks: List of [price, size] for asks
            price: Current market price
            
        Returns:
            Analysis results
        """
        analysis = {
            'whale_walls': [],
            'new_whale_orders': [],
            'removed_whale_orders': [],
            'total_whale_bid_volume': 0.0,
            'total_whale_ask_volume': 0.0,
            'dominant_side': None,
        }
        
        # Convert to tuples for comparison
        current_bids = [(float(b[0]), float(b[1])) for b in bids]
        current_asks = [(float(a[0]), float(a[1])) for a in asks]
        
        # Detect whale walls (large orders)
        for bid_price, bid_size in current_bids:
            if bid_size >= self.whale_threshold:
                distance_pct = ((price - bid_price) / price) * 100
                analysis['whale_walls'].append({
                    'side': 'bid',
                    'price': bid_price,
                    'size': bid_size,
                    'distance_from_price_pct': distance_pct,
                })
                analysis['total_whale_bid_volume'] += bid_size
        
        for ask_price, ask_size in current_asks:
            if ask_size >= self.whale_threshold:
                distance_pct = ((ask_price - price) / price) * 100
                analysis['whale_walls'].append({
                    'side': 'ask',
                    'price': ask_price,
                    'size': ask_size,
                    'distance_from_price_pct': distance_pct,
                })
                analysis['total_whale_ask_volume'] += ask_size
        
        # Detect new whale orders (appeared since last update)
        if self.previous_bids:
            new_whale_bids = self._find_new_whale_orders(
                current_bids, self.previous_bids, 'bid'
            )
            analysis['new_whale_orders'].extend(new_whale_bids)
        
        if self.previous_asks:
            new_whale_asks = self._find_new_whale_orders(
                current_asks, self.previous_asks, 'ask'
            )
            analysis['new_whale_orders'].extend(new_whale_asks)
        
        # Detect removed whale orders
        if self.previous_bids:
            removed_bids = self._find_removed_whale_orders(
                self.previous_bids, current_bids, 'bid'
            )
            analysis['removed_whale_orders'].extend(removed_bids)
        
        if self.previous_asks:
            removed_asks = self._find_removed_whale_orders(
                self.previous_asks, current_asks, 'ask'
            )
            analysis['removed_whale_orders'].extend(removed_asks)
        
        # Determine dominant side
        if analysis['total_whale_bid_volume'] > analysis['total_whale_ask_volume'] * 1.5:
            analysis['dominant_side'] = 'bid'
        elif analysis['total_whale_ask_volume'] > analysis['total_whale_bid_volume'] * 1.5:
            analysis['dominant_side'] = 'ask'
        else:
            analysis['dominant_side'] = 'balanced'
        
        # Update state
        self.previous_bids = current_bids
        self.previous_asks = current_asks
        
        # Log significant events
        if analysis['new_whale_orders']:
            for order in analysis['new_whale_orders']:
                logger.warning(
                    f"🐋 NEW WHALE ORDER: {order['side'].upper()} "
                    f"{order['size']:.2f} BTC @ ${order['price']:,.2f}"
                )
        
        if analysis['removed_whale_orders']:
            for order in analysis['removed_whale_orders']:
                logger.warning(
                    f"🐋 WHALE ORDER REMOVED: {order['side'].upper()} "
                    f"{order['size']:.2f} BTC @ ${order['price']:,.2f}"
                )
        
        return analysis
    
    def _find_new_whale_orders(
        self,
        current: List[Tuple[float, float]],
        previous: List[Tuple[float, float]],
        side: str
    ) -> List[Dict]:
        """Find whale orders that appeared since last update"""
        new_orders = []
        previous_dict = {price: size for price, size in previous}
        
        for price, size in current:
            if size >= self.whale_threshold:
                # New order or size increased significantly
                if price not in previous_dict or size > previous_dict[price] * 1.5:
                    new_orders.append({
                        'side': side,
                        'price': price,
                        'size': size,
                        'previous_size': previous_dict.get(price, 0),
                    })
        
        return new_orders
    
    def _find_removed_whale_orders(
        self,
        previous: List[Tuple[float, float]],
        current: List[Tuple[float, float]],
        side: str
    ) -> List[Dict]:
        """Find whale orders that were removed"""
        removed = []
        current_dict = {price: size for price, size in current}
        
        for price, size in previous:
            if size >= self.whale_threshold:
                # Order removed or size decreased significantly
                if price not in current_dict or current_dict[price] < size * 0.5:
                    removed.append({
                        'side': side,
                        'price': price,
                        'size': size,
                        'current_size': current_dict.get(price, 0),
                    })
        
        return removed