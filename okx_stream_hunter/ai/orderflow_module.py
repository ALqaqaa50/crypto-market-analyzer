"""
PROMETHEUS AI BRAIN v7 (OMEGA EDITION)
Orderflow Neural Layer

Analyzes real-time orderflow from orderbook and trades:
- Bid/ask volume analysis
- Spoofing detection
- Aggressive buying/selling
- Pressure and imbalance
- Iceberg order detection
- Volume clustering
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderflowOutput:
    """Output from Orderflow analysis"""
    signal: str  # 'bullish', 'bearish', 'neutral'
    confidence: float
    metrics: Dict[str, float]
    alerts: List[str]  # Detected anomalies


class OrderflowModule:
    """Real-time orderflow intelligence"""
    
    def __init__(self, config):
        self.config = config
        self.trades_micro = deque(maxlen=1000)
        self.trades_short = deque(maxlen=5000)
        self.trades_medium = deque(maxlen=20000)
        self.orderbook_snapshots = deque(maxlen=100)
        self.cancelled_orders = deque(maxlen=500)
        
        logger.info("Initialized Orderflow Neural Layer")
    
    def process_trade(self, trade: Dict):
        """Process incoming trade"""
        trade['timestamp'] = datetime.now()
        self.trades_micro.append(trade)
        self.trades_short.append(trade)
        self.trades_medium.append(trade)
    
    def process_orderbook(self, orderbook: Dict):
        """Process orderbook snapshot"""
        orderbook['timestamp'] = datetime.now()
        self.orderbook_snapshots.append(orderbook)
    
    def process_cancelled_order(self, order: Dict):
        """Track cancelled orders for spoof detection"""
        order['timestamp'] = datetime.now()
        self.cancelled_orders.append(order)
    
    def analyze(self) -> OrderflowOutput:
        """Comprehensive orderflow analysis"""
        metrics = {}
        alerts = []
        
        # 1. Buy/Sell pressure
        buy_pressure, sell_pressure = self._calculate_pressure()
        metrics['buy_pressure'] = buy_pressure
        metrics['sell_pressure'] = sell_pressure
        metrics['net_pressure'] = buy_pressure - sell_pressure
        
        # 2. Spoofing detection
        spoof_score = self._detect_spoofing()
        metrics['spoof_score'] = spoof_score
        if spoof_score > self.config.spoofing_threshold:
            alerts.append(f"SPOOFING_DETECTED: {spoof_score:.2f}")
        
        # 3. Aggressive orders
        aggr_buy, aggr_sell = self._detect_aggressive_orders()
        metrics['aggressive_buy'] = aggr_buy
        metrics['aggressive_sell'] = aggr_sell
        
        # 4. Orderbook imbalance
        imbalance = self._calculate_orderbook_imbalance()
        metrics['orderbook_imbalance'] = imbalance
        
        # 5. Iceberg detection
        iceberg_score = self._detect_icebergs()
        metrics['iceberg_score'] = iceberg_score
        if iceberg_score > self.config.iceberg_detection_threshold:
            alerts.append(f"ICEBERG_DETECTED: {iceberg_score:.2f}")
        
        # 6. Absorption
        absorption = self._detect_absorption()
        metrics['absorption'] = absorption
        if absorption > self.config.absorption_threshold:
            alerts.append(f"ABSORPTION: {absorption:.2f}")
        
        # 7. Volume clustering
        clusters = self._detect_volume_clusters()
        metrics['volume_clusters'] = len(clusters)
        
        # Determine signal
        signal, confidence = self._determine_signal(metrics)
        
        return OrderflowOutput(
            signal=signal,
            confidence=confidence,
            metrics=metrics,
            alerts=alerts
        )
    
    def _calculate_pressure(self) -> Tuple[float, float]:
        """Calculate buy/sell pressure from recent trades"""
        if not self.trades_short:
            return 0.5, 0.5
        
        cutoff = datetime.now() - timedelta(seconds=self.config.short_window_seconds)
        recent_trades = [t for t in self.trades_short if t['timestamp'] > cutoff]
        
        buy_volume = sum(t.get('size', 0) for t in recent_trades if t.get('side') == 'buy')
        sell_volume = sum(t.get('size', 0) for t in recent_trades if t.get('side') == 'sell')
        
        total = buy_volume + sell_volume
        if total == 0:
            return 0.5, 0.5
        
        return buy_volume / total, sell_volume / total
    
    def _detect_spoofing(self) -> float:
        """Detect spoofing patterns"""
        if not self.cancelled_orders or not self.trades_short:
            return 0.0
        
        cutoff = datetime.now() - timedelta(seconds=self.config.short_window_seconds)
        recent_cancels = [o for o in self.cancelled_orders if o['timestamp'] > cutoff]
        recent_trades = [t for t in self.trades_short if t['timestamp'] > cutoff]
        
        cancel_volume = sum(o.get('size', 0) for o in recent_cancels)
        trade_volume = sum(t.get('size', 0) for t in recent_trades)
        
        if trade_volume == 0:
            return 0.0
        
        # High cancel-to-trade ratio indicates spoofing
        return min(cancel_volume / (trade_volume + 1), 1.0)
    
    def _detect_aggressive_orders(self) -> Tuple[float, float]:
        """Detect aggressive market orders"""
        if not self.trades_micro:
            return 0.0, 0.0
        
        cutoff = datetime.now() - timedelta(seconds=self.config.micro_window_seconds)
        recent = [t for t in self.trades_micro if t['timestamp'] > cutoff]
        
        # Aggressive = market orders that cross spread
        aggr_buy = sum(1 for t in recent if t.get('side') == 'buy' and t.get('aggressive', False))
        aggr_sell = sum(1 for t in recent if t.get('side') == 'sell' and t.get('aggressive', False))
        
        total = len(recent) if recent else 1
        return aggr_buy / total, aggr_sell / total
    
    def _calculate_orderbook_imbalance(self) -> float:
        """Calculate bid/ask imbalance"""
        if not self.orderbook_snapshots:
            return 0.5
        
        latest = self.orderbook_snapshots[-1]
        
        bid_volume = sum(level[1] for level in latest.get('bids', [])[:10])
        ask_volume = sum(level[1] for level in latest.get('asks', [])[:10])
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.5
        
        return bid_volume / total
    
    def _detect_icebergs(self) -> float:
        """Detect hidden iceberg orders"""
        if len(self.orderbook_snapshots) < 2:
            return 0.0
        
        # Compare visible vs executed volume
        # Iceberg: large trades with small visible orderbook depth
        recent_trades_volume = sum(t.get('size', 0) for t in list(self.trades_micro)[-20:])
        
        latest_ob = self.orderbook_snapshots[-1]
        visible_depth = sum(level[1] for level in latest_ob.get('bids', [])[:5])
        visible_depth += sum(level[1] for level in latest_ob.get('asks', [])[:5])
        
        if visible_depth == 0:
            return 0.0
        
        # High execution vs low visibility = iceberg
        return min(recent_trades_volume / (visible_depth + 1), 1.0)
    
    def _detect_absorption(self) -> float:
        """Detect volume absorption (large volume, small price move)"""
        if len(self.trades_short) < 10:
            return 0.0
        
        cutoff = datetime.now() - timedelta(seconds=self.config.short_window_seconds)
        recent = [t for t in self.trades_short if t['timestamp'] > cutoff]
        
        if not recent:
            return 0.0
        
        total_volume = sum(t.get('size', 0) for t in recent)
        price_change = abs(recent[-1].get('price', 0) - recent[0].get('price', 1)) / recent[0].get('price', 1)
        
        # High volume with low price change = absorption
        if price_change < 0.001 and total_volume > 0:  # Less than 0.1% move
            return min(total_volume / 1000, 1.0)
        
        return 0.0
    
    def _detect_volume_clusters(self) -> List[Dict]:
        """Detect rapid volume clusters"""
        clusters = []
        
        if len(self.trades_micro) < self.config.volume_cluster_min_trades:
            return clusters
        
        window_size = self.config.volume_cluster_time_window
        trades = list(self.trades_micro)
        
        i = 0
        while i < len(trades) - self.config.volume_cluster_min_trades:
            cluster_trades = []
            start_time = trades[i]['timestamp']
            
            for j in range(i, len(trades)):
                if (trades[j]['timestamp'] - start_time).total_seconds() <= window_size:
                    cluster_trades.append(trades[j])
                else:
                    break
            
            if len(cluster_trades) >= self.config.volume_cluster_min_trades:
                total_volume = sum(t.get('size', 0) for t in cluster_trades)
                clusters.append({
                    'start_time': start_time,
                    'trade_count': len(cluster_trades),
                    'total_volume': total_volume
                })
                i += len(cluster_trades)
            else:
                i += 1
        
        return clusters
    
    def _determine_signal(self, metrics: Dict[str, float]) -> Tuple[str, float]:
        """Determine overall orderflow signal"""
        # Weighted scoring
        score = 0.0
        confidence = 0.0
        
        # Net pressure (positive = bullish, negative = bearish)
        net_pressure = metrics.get('net_pressure', 0)
        score += net_pressure * 0.4
        
        # Orderbook imbalance (>0.5 = more bids = bullish)
        imbalance = metrics.get('orderbook_imbalance', 0.5) - 0.5
        score += imbalance * 0.3
        
        # Aggressive orders
        aggr_buy = metrics.get('aggressive_buy', 0)
        aggr_sell = metrics.get('aggressive_sell', 0)
        score += (aggr_buy - aggr_sell) * 0.3
        
        # Reduce confidence if spoofing detected
        spoof_penalty = 1.0 - metrics.get('spoof_score', 0) * 0.5
        
        # Determine signal
        if score > 0.15:
            signal = 'bullish'
            confidence = min(abs(score) * 2, 1.0) * spoof_penalty
        elif score < -0.15:
            signal = 'bearish'
            confidence = min(abs(score) * 2, 1.0) * spoof_penalty
        else:
            signal = 'neutral'
            confidence = 0.5 * spoof_penalty
        
        return signal, confidence
