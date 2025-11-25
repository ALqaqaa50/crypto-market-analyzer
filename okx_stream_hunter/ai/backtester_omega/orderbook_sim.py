import numpy as np
import logging
from typing import Dict, List, Tuple
from collections import deque
import time

logger = logging.getLogger(__name__)


class OrderbookSimulator:
    def __init__(self, depth: int = 20, latency_ms: float = 50.0, spread_percent: float = 0.001):
        self.depth = depth
        self.latency_ms = latency_ms
        self.spread_percent = spread_percent
        
        self.bids = []
        self.asks = []
        
        self.pending_orders = deque()
        self.filled_orders = []
        
        self.current_price = None
        self.last_update_time = time.time()
        
        self.total_volume = 0.0
        self.total_trades = 0
        
        logger.info(f"OrderbookSimulator initialized: depth={depth}, latency={latency_ms}ms")
    
    def update_orderbook(self, mid_price: float, volume: float):
        spread = mid_price * self.spread_percent
        
        self.bids = []
        self.asks = []
        
        for i in range(self.depth):
            bid_price = mid_price - spread * (i + 1) / self.depth
            ask_price = mid_price + spread * (i + 1) / self.depth
            
            level_volume = volume * (1.0 / (i + 1)**1.5)
            
            self.bids.append({
                'price': bid_price,
                'volume': level_volume,
                'orders': max(1, int(level_volume / 0.01))
            })
            
            self.asks.append({
                'price': ask_price,
                'volume': level_volume,
                'orders': max(1, int(level_volume / 0.01))
            })
        
        self.current_price = mid_price
        self.last_update_time = time.time()
        
        self._process_pending_orders()
    
    def place_order(self, side: str, size: float, order_type: str = 'market') -> Dict:
        order = {
            'id': len(self.filled_orders) + len(self.pending_orders),
            'side': side,
            'size': size,
            'type': order_type,
            'timestamp': time.time(),
            'status': 'pending'
        }
        
        self.pending_orders.append(order)
        
        return order
    
    def _process_pending_orders(self):
        processed = []
        
        while self.pending_orders:
            order = self.pending_orders.popleft()
            
            time_elapsed = (time.time() - order['timestamp']) * 1000
            
            if time_elapsed < self.latency_ms:
                self.pending_orders.append(order)
                break
            
            fill_result = self._fill_order(order)
            
            order['status'] = 'filled' if fill_result['filled'] else 'rejected'
            order['fill_price'] = fill_result.get('price', None)
            order['fill_size'] = fill_result.get('size', 0)
            order['slippage'] = fill_result.get('slippage', 0)
            
            self.filled_orders.append(order)
            processed.append(order)
        
        return processed
    
    def _fill_order(self, order: Dict) -> Dict:
        side = order['side']
        size = order['size']
        
        if side == 'buy':
            available_liquidity = sum(ask['volume'] for ask in self.asks)
        else:
            available_liquidity = sum(bid['volume'] for bid in self.bids)
        
        if size > available_liquidity:
            return {'filled': False, 'reason': 'insufficient_liquidity'}
        
        filled_size = 0
        weighted_price = 0
        
        levels = self.asks if side == 'buy' else self.bids
        
        for level in levels:
            if filled_size >= size:
                break
            
            take_size = min(size - filled_size, level['volume'])
            
            weighted_price += level['price'] * take_size
            filled_size += take_size
        
        avg_fill_price = weighted_price / filled_size
        
        slippage = (avg_fill_price - self.current_price) / self.current_price
        if side == 'sell':
            slippage = -slippage
        
        self.total_volume += filled_size
        self.total_trades += 1
        
        return {
            'filled': True,
            'price': avg_fill_price,
            'size': filled_size,
            'slippage': slippage
        }
    
    def get_best_bid_ask(self) -> Tuple[float, float]:
        if not self.bids or not self.asks:
            return None, None
        
        return self.bids[0]['price'], self.asks[0]['price']
    
    def get_orderbook_state(self) -> Dict:
        bid_volume = sum(bid['volume'] for bid in self.bids)
        ask_volume = sum(ask['volume'] for ask in self.asks)
        
        return {
            'mid_price': self.current_price,
            'best_bid': self.bids[0]['price'] if self.bids else None,
            'best_ask': self.asks[0]['price'] if self.asks else None,
            'spread': (self.asks[0]['price'] - self.bids[0]['price']) if self.bids and self.asks else None,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8),
            'total_trades': self.total_trades,
            'total_volume': self.total_volume
        }
