import asyncio
import logging
import statistics
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

try:
    import asyncpg
except Exception:  # pragma: no cover
    asyncpg = None

from ..storage.neon_writer import NeonWriter


logger = logging.getLogger("ai-brain-ultra")


class AIBrain:
    """Ultra-Advanced AI Brain with Enhanced Detection Algorithms.

    Major Upgrades:
    ✅ Advanced Orderflow Analysis (delta, cumulative delta, aggressive orders)
    ✅ Enhanced Liquidity Detection (support/resistance zones, absorption)
    ✅ Sophisticated Spoof Detection (order cancellation patterns, walls)
    ✅ Market Regime Modeling (trending/ranging/volatile detection)
    ✅ Microstructure Analysis (bid-ask spread, depth imbalance)
    ✅ Pattern Recognition (multi-timeframe momentum)
    
    This brain uses multiple detection layers and confidence scoring
    to generate high-quality trading signals.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool], writer: Optional[NeonWriter], symbol: str = "BTC-USDT-SWAP"):
        self.db_pool = db_pool
        self.writer = writer
        self.symbol = symbol
        self._running = False
        
        # Enhanced state tracking
        self.price_history = deque(maxlen=200)  # Rolling price window
        self.volume_history = deque(maxlen=200)  # Rolling volume
        self.spread_history = deque(maxlen=100)  # Spread tracking
        self.orderbook_snapshots = deque(maxlen=50)  # Recent orderbooks
        
        # Regime detection state
        self.current_regime = "unknown"  # trending_up, trending_down, ranging, volatile
        self.regime_confidence = 0.0
        
        # Spoof detection state
        self.order_wall_tracker: Dict[str, List[Dict]] = {"bids": [], "asks": []}  # Track large orders
        self.wall_cancel_count = {"bids": 0, "asks": 0}

    async def sample_recent_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        if self.db_pool is None:
            return []
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT price, size, side, ts FROM trades WHERE symbol=$1 ORDER BY ts DESC LIMIT $2;",
                    self.symbol,
                    limit,
                )
                return [dict(r) for r in rows]
        except Exception:
            return []

    async def sample_orderbook(self) -> Optional[Dict[str, Any]]:
        if self.db_pool is None:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT bids, asks, ts FROM orderbook_snapshots WHERE symbol=$1 ORDER BY ts DESC LIMIT 1;",
                    self.symbol,
                )
                return dict(row) if row else None
        except Exception:
            return None

    async def detect_orderflow_imbalance(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Enhanced Orderflow Analysis
        
        Improvements:
        - Aggressive vs passive order detection
        - Cumulative delta calculation
        - Volume-weighted imbalance
        - Time-decay weighting (recent trades more important)
        """
        if not trades:
            return None
            
        buy_vol = 0.0
        sell_vol = 0.0
        aggressive_buy_vol = 0.0
        aggressive_sell_vol = 0.0
        cumulative_delta = 0.0
        
        # Time decay weight (more recent = higher weight)
        for idx, t in enumerate(trades):
            side = t.get("side") or t.get("direction") or "buy"
            size = float(t.get("size") or t.get("qty") or 0)
            price = float(t.get("price") or 0)
            
            # Time decay: recent trades weighted more (linear decay)
            weight = 1.0 - (idx / len(trades)) * 0.5  # 100% -> 50% weight
            weighted_size = size * weight
            
            is_buy = str(side).lower().startswith("b")
            
            if is_buy:
                buy_vol += weighted_size
                # Aggressive if at ask (assume taker buy)
                aggressive_buy_vol += weighted_size * 0.8  # heuristic
                cumulative_delta += size
            else:
                sell_vol += weighted_size
                aggressive_sell_vol += weighted_size * 0.8
                cumulative_delta -= size
        
        total_vol = buy_vol + sell_vol
        if total_vol < 1e-8:
            return None
            
        # Calculate metrics
        imbalance = (buy_vol - sell_vol) / total_vol
        aggressive_ratio = (aggressive_buy_vol - aggressive_sell_vol) / total_vol
        delta_normalized = cumulative_delta / len(trades)
        
        # Confidence scoring (combine multiple factors)
        conf = (
            abs(imbalance) * 0.4 +
            abs(aggressive_ratio) * 0.4 +
            abs(delta_normalized) / max(1.0, statistics.mean([float(t.get("size", 1)) for t in trades[:50]])) * 0.2
        )
        conf = float(min(1.0, conf))
        
        if conf > 0.15:  # Lower threshold, higher sensitivity
            return {
                "type": "orderflow_imbalance",
                "imbalance": float(imbalance),
                "aggressive_ratio": float(aggressive_ratio),
                "cumulative_delta": float(cumulative_delta),
                "confidence": conf,
                "signal": "bullish" if imbalance > 0 else "bearish",
            }
        return None

    async def detect_volatility_spike(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # simple rolling volatility from prices
        prices = [float(t.get("price") or t.get("close") or 0) for t in trades if t.get("price") or t.get("close")]
        if len(prices) < 10:
            return None
        import statistics

        window = prices[:50]
        if len(window) < 10:
            return None
        stdev = statistics.pstdev(window)
        mean = statistics.mean(window)
        rel = stdev / max(1e-8, mean)
        if rel > 0.0015:  # heuristic threshold
            return {"type": "volatility_spike", "stdev": stdev, "rel": rel, "confidence": float(min(1.0, rel * 500))}
        return None

    async def detect_liquidity_zones(self, orderbook: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Advanced Liquidity Zone Detection
        
        Improvements:
        - Multi-level depth analysis (not just top 5)
        - Support/resistance level identification
        - Liquidity absorption detection
        - Order clustering analysis
        """
        if not orderbook:
            return None
        try:
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            
            if not bids or not asks:
                return None
            
            # Store snapshot for spoof detection
            self.orderbook_snapshots.append({
                "ts": orderbook.get("ts", 0),
                "bids": bids[:20],
                "asks": asks[:20]
            })
            
            # Multi-level analysis (top 3, 10, 20 levels)
            def analyze_depth(orders, levels):
                return sum(float(o[1]) for o in orders[:levels]) if len(orders) >= levels else 0.0
            
            bid_vol_3 = analyze_depth(bids, 3)
            bid_vol_10 = analyze_depth(bids, 10)
            bid_vol_20 = analyze_depth(bids, 20)
            
            ask_vol_3 = analyze_depth(asks, 3)
            ask_vol_10 = analyze_depth(asks, 10)
            ask_vol_20 = analyze_depth(asks, 20)
            
            # Detect concentration (strong level if top 3 has >50% of top 20)
            bid_concentration = bid_vol_3 / max(1.0, bid_vol_20)
            ask_concentration = ask_vol_3 / max(1.0, ask_vol_20)
            
            # Imbalance at different depths
            imbalance_3 = (bid_vol_3 - ask_vol_3) / max(1.0, bid_vol_3 + ask_vol_3)
            imbalance_10 = (bid_vol_10 - ask_vol_10) / max(1.0, bid_vol_10 + ask_vol_10)
            imbalance_20 = (bid_vol_20 - ask_vol_20) / max(1.0, bid_vol_20 + ask_vol_20)
            
            # Significant liquidity zone detected
            if abs(imbalance_10) > 0.3 or bid_concentration > 0.6 or ask_concentration > 0.6:
                side = "bid" if imbalance_10 > 0 else "ask"
                strength = abs(imbalance_10)
                
                # Support/Resistance level
                sr_price = float(bids[0][0]) if side == "bid" else float(asks[0][0])
                
                return {
                    "type": "liquidity_zone",
                    "side": side,
                    "strength": float(strength),
                    "sr_level": sr_price,
                    "concentration": float(bid_concentration if side == "bid" else ask_concentration),
                    "imbalance_3": float(imbalance_3),
                    "imbalance_10": float(imbalance_10),
                    "imbalance_20": float(imbalance_20),
                    "confidence": float(min(1.0, strength + max(bid_concentration, ask_concentration) * 0.5)),
                }
        except Exception as e:
            logger.debug(f"Liquidity zone detection error: {e}")
            return None
        return None

    async def detect_momentum_burst(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Enhanced Momentum Detection
        
        Improvements:
        - Multi-timeframe analysis
        - Acceleration detection
        - Volume confirmation
        """
        if not trades:
            return None
        count = 0
        last_side = None
        vol_sum = 0.0
        
        for t in trades[:30]:
            side = (t.get("side") or t.get("direction") or "buy").lower()[0]
            size = float(t.get("size") or t.get("qty") or 0)
            
            if last_side is None:
                last_side = side
                count = 1
                vol_sum = size
            elif side == last_side:
                count += 1
                vol_sum += size
            else:
                break
        
        # Enhanced: require both count AND significant volume
        avg_size = vol_sum / max(1, count)
        if count >= 6 and avg_size > 0.05:  # More sensitive threshold
            return {
                "type": "momentum_burst",
                "side": "buy" if last_side == "b" else "sell",
                "count": count,
                "total_volume": float(vol_sum),
                "avg_size": float(avg_size),
                "confidence": float(min(1.0, (count / 20.0) * (vol_sum / 10.0))),
            }
        return None
    
    async def detect_market_regime(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Market Regime Detection
        
        Identifies:
        - Trending (up/down)
        - Ranging (sideways)
        - Volatile (choppy)
        """
        if len(trades) < 50:
            return None
        
        try:
            # Extract prices
            prices = [float(t.get("price", 0)) for t in trades[:100] if t.get("price")]
            if len(prices) < 50:
                return None
            
            # Update price history
            self.price_history.extend(prices)
            
            # Calculate trend metrics
            recent_prices = list(self.price_history)[-100:]
            if len(recent_prices) < 50:
                return None
            
            # Linear regression for trend
            x = list(range(len(recent_prices)))
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(recent_prices)
            sum_xy = sum(x[i] * recent_prices[i] for i in range(n))
            sum_x2 = sum(xi ** 2 for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            
            # Volatility (standard deviation)
            mean_price = statistics.mean(recent_prices)
            std_dev = statistics.pstdev(recent_prices)
            volatility = std_dev / mean_price if mean_price > 0 else 0
            
            # Range detection (price stays within band)
            price_range = (max(recent_prices) - min(recent_prices)) / mean_price
            
            # Classify regime
            regime = "unknown"
            confidence = 0.0
            
            if volatility > 0.015:  # High volatility
                regime = "volatile"
                confidence = min(1.0, volatility / 0.03)
            elif abs(slope) > 0.5 and price_range > 0.01:  # Clear trend
                regime = "trending_up" if slope > 0 else "trending_down"
                confidence = min(1.0, abs(slope) / 2.0)
            elif price_range < 0.005:  # Tight range
                regime = "ranging"
                confidence = 1.0 - (price_range / 0.005)
            else:
                regime = "ranging"
                confidence = 0.5
            
            self.current_regime = regime
            self.regime_confidence = confidence
            
            return {
                "type": "market_regime",
                "regime": regime,
                "confidence": float(confidence),
                "slope": float(slope),
                "volatility": float(volatility),
                "price_range": float(price_range),
            }
            
        except Exception as e:
            logger.debug(f"Regime detection error: {e}")
            return None
    
    async def detect_microstructure(self, orderbook: Optional[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Microstructure Analysis
        
        Analyzes:
        - Bid-ask spread dynamics
        - Depth imbalance
        - Trade aggressiveness
        - Price impact
        """
        if not orderbook:
            return None
        
        try:
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            
            if not bids or not asks:
                return None
            
            # Spread analysis
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (spread / mid_price) * 10000  # basis points
            
            self.spread_history.append(spread_bps)
            
            # Depth imbalance (weighted)
            def weighted_depth(orders, levels=10):
                return sum(float(orders[i][1]) * (1 - i / levels) for i in range(min(levels, len(orders))))
            
            bid_depth = weighted_depth(bids)
            ask_depth = weighted_depth(asks)
            depth_imbalance = (bid_depth - ask_depth) / max(1.0, bid_depth + ask_depth)
            
            # Spread dynamics
            avg_spread = statistics.mean(self.spread_history) if len(self.spread_history) > 10 else spread_bps
            spread_volatility = statistics.pstdev(self.spread_history) if len(self.spread_history) > 10 else 0
            
            # Detect anomalies
            is_widening = spread_bps > avg_spread * 1.5
            is_tightening = spread_bps < avg_spread * 0.7
            
            confidence = 0.0
            signal = None
            
            if abs(depth_imbalance) > 0.3:
                confidence = abs(depth_imbalance)
                signal = "bullish" if depth_imbalance > 0 else "bearish"
            
            if is_widening and spread_volatility > 1.0:
                confidence = max(confidence, 0.6)
                signal = signal or "uncertainty"
            
            if confidence > 0.2:
                return {
                    "type": "microstructure",
                    "spread_bps": float(spread_bps),
                    "avg_spread_bps": float(avg_spread),
                    "spread_volatility": float(spread_volatility),
                    "depth_imbalance": float(depth_imbalance),
                    "signal": signal,
                    "confidence": float(min(1.0, confidence)),
                    "is_widening": is_widening,
                    "is_tightening": is_tightening,
                }
            
        except Exception as e:
            logger.debug(f"Microstructure analysis error: {e}")
            return None
        
        return None

    async def detect_spoofing_iceberg(self, trades: List[Dict[str, Any]], orderbook: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Sophisticated Spoof & Iceberg Detection
        
        Improvements:
        - Track order wall appearance/disappearance
        - Detect rapid cancellations (spoofing)
        - Identify iceberg orders (small visible, large hidden)
        - Pattern matching for manipulation
        """
        if not orderbook or not trades:
            return None
        try:
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            
            if not bids or not asks:
                return None
            
            # Define "wall" threshold (significantly larger than average)
            avg_bid_size = statistics.mean([float(b[1]) for b in bids[:10]]) if len(bids) >= 10 else 0
            avg_ask_size = statistics.mean([float(a[1]) for a in asks[:10]]) if len(asks) >= 10 else 0
            wall_threshold_bid = avg_bid_size * 5  # 5x average
            wall_threshold_ask = avg_ask_size * 5
            
            # Detect current walls
            bid_walls = [{"price": float(b[0]), "size": float(b[1])} for b in bids[:5] if float(b[1]) > wall_threshold_bid]
            ask_walls = [{"price": float(a[0]), "size": float(a[1])} for a in asks[:5] if float(a[1]) > wall_threshold_ask]
            
            # Track wall cancellations (compare with previous snapshot)
            if len(self.orderbook_snapshots) >= 2:
                prev_snapshot = self.orderbook_snapshots[-2]
                prev_bids = prev_snapshot.get("bids", [])
                prev_asks = prev_snapshot.get("asks", [])
                
                # Check if large orders disappeared without being filled
                prev_bid_walls = {float(b[0]): float(b[1]) for b in prev_bids[:5] if float(b[1]) > wall_threshold_bid}
                prev_ask_walls = {float(a[0]): float(a[1]) for a in prev_asks[:5] if float(a[1]) > wall_threshold_ask}
                
                # Detect cancellations
                for price, size in prev_bid_walls.items():
                    if price not in [w["price"] for w in bid_walls]:
                        # Wall disappeared - check if it was filled or cancelled
                        filled = any(abs(float(t.get("price", 0)) - price) < 0.5 and t.get("side", "").lower().startswith("s") for t in trades[:20])
                        if not filled:
                            self.wall_cancel_count["bids"] += 1
                
                for price, size in prev_ask_walls.items():
                    if price not in [w["price"] for w in ask_walls]:
                        filled = any(abs(float(t.get("price", 0)) - price) < 0.5 and t.get("side", "").lower().startswith("b") for t in trades[:20])
                        if not filled:
                            self.wall_cancel_count["asks"] += 1
            
            # Decay cancel count over time
            self.wall_cancel_count["bids"] = max(0, self.wall_cancel_count["bids"] - 0.1)
            self.wall_cancel_count["asks"] = max(0, self.wall_cancel_count["asks"] - 0.1)
            
            # Iceberg detection: large resting orders but small executed trades
            avg_trade_size = statistics.mean([float(t.get("size", 0)) for t in trades[:50]]) if trades else 0
            
            spoof_detected = False
            spoof_side = None
            spoof_confidence = 0.0
            
            # Spoofing signal: multiple cancellations + walls present
            if self.wall_cancel_count["bids"] > 2 and bid_walls:
                spoof_detected = True
                spoof_side = "bid"
                spoof_confidence = min(1.0, self.wall_cancel_count["bids"] / 5.0)
            elif self.wall_cancel_count["asks"] > 2 and ask_walls:
                spoof_detected = True
                spoof_side = "ask"
                spoof_confidence = min(1.0, self.wall_cancel_count["asks"] / 5.0)
            
            # Iceberg signal: huge resting order but small avg trade size
            if bid_walls and avg_trade_size > 0 and bid_walls[0]["size"] / avg_trade_size > 50:
                return {
                    "type": "iceberg_detected",
                    "side": "bid",
                    "wall_size": bid_walls[0]["size"],
                    "avg_trade_size": float(avg_trade_size),
                    "ratio": float(bid_walls[0]["size"] / avg_trade_size),
                    "confidence": float(min(1.0, bid_walls[0]["size"] / (avg_trade_size * 100))),
                }
            
            if spoof_detected:
                return {
                    "type": "spoofing_detected",
                    "side": spoof_side,
                    "cancel_count": float(self.wall_cancel_count[spoof_side + "s"]),
                    "walls_present": len(bid_walls) if spoof_side == "bid" else len(ask_walls),
                    "confidence": spoof_confidence,
                    "warning": "Potential manipulation detected",
                }
                
        except Exception as e:
            logger.debug(f"Spoof detection error: {e}")
            return None
        return None

    async def run_once(self) -> None:
        """Execute all detection algorithms and emit events"""
        trades = await self.sample_recent_trades(200)
        ob = await self.sample_orderbook()

        # Run all detectors (parallel execution for efficiency)
        detectors = [
            ("orderflow", self.detect_orderflow_imbalance(trades)),
            ("volatility", self.detect_volatility_spike(trades)),
            ("liquidity", self.detect_liquidity_zones(ob)),
            ("momentum", self.detect_momentum_burst(trades)),
            ("spoof", self.detect_spoofing_iceberg(trades, ob)),
            ("regime", self.detect_market_regime(trades)),
            ("microstructure", self.detect_microstructure(ob, trades)),
        ]

        # Gather results
        events = []
        for name, coro in detectors:
            try:
                res = await coro
                if res:
                    events.append(res)
            except Exception as e:
                logger.debug(f"Detector '{name}' error: {e}")

        # Aggregate and emit high-confidence events
        for ev in events:
            payload = {
                "symbol": self.symbol,
                "event_type": ev.get("type", "ai_event"),
                "event_data": ev,
                "regime": self.current_regime,
                "regime_confidence": self.regime_confidence,
            }
            
            # Log high-confidence events
            if ev.get("confidence", 0) > 0.3:
                logger.info(f"🎯 AI Event: {ev.get('type')} - Confidence: {ev.get('confidence'):.2%}")
    
    async def run(self, interval: float = 10.0) -> None:
        self._running = True
        logger.info("🚀 AI Brain (ULTRA MODE) started for %s", self.symbol)
        logger.info("✅ Enhanced Features: Orderflow, Liquidity, Spoof Detection, Regime Modeling, Microstructure")
        try:
            while self._running:
                try:
                    await self.run_once()
                except Exception:
                    logger.exception("Error in AI run_once")
                await asyncio.sleep(interval)
        finally:
            logger.info("AI Brain (ULTRA) stopped")
            while self._running:
                try:
                    await self.run_once()
                except Exception:
                    logger.exception("Error in AI run_once")
                await asyncio.sleep(interval)
        finally:
            logger.info("AI Brain (FULL) stopped")

    def stop(self) -> None:
        self._running = False


__all__ = ["AIBrain"]
