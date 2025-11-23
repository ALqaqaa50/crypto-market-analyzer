import asyncio
import logging
from typing import Optional, Dict, Any, List

try:
    import asyncpg
except Exception:  # pragma: no cover
    asyncpg = None

from ..storage.neon_writer import NeonWriter


logger = logging.getLogger("ai-brain-full")


class AIBrain:
    """Full-mode AI brain (heuristic detectors + event writer).

    This is a modular, extensible implementation that executes a set of
    detectors over recent market data (from DB when available) and
    emits `market_events` via `NeonWriter`.

    Detectors implemented (heuristics):
      - orderflow imbalance
      - volatility spike
      - basic liquidity zone detection
      - momentum burst
      - simple spoof/iceberg hinting

    Note: These are heuristic detectors meant as a scaffold — replace with
    ML models or more sophisticated logic as needed.
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool], writer: Optional[NeonWriter], symbol: str = "BTC-USDT-SWAP"):
        self.db_pool = db_pool
        self.writer = writer
        self.symbol = symbol
        self._running = False

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
        if not trades:
            return None
        buy_vol = 0.0
        sell_vol = 0.0
        for t in trades:
            side = t.get("side") or t.get("direction") or "buy"
            size = float(t.get("size") or t.get("qty") or 0)
            if str(side).lower().startswith("b"):
                buy_vol += size
            else:
                sell_vol += size

        imbalance = (buy_vol - sell_vol) / max(1.0, buy_vol + sell_vol)
        conf = abs(imbalance)
        if conf > 0.2:
            return {
                "type": "orderflow_imbalance",
                "imbalance": imbalance,
                "confidence": float(min(1.0, conf * 1.2)),
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
        if not orderbook:
            return None
        try:
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            top_bids = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
            top_asks = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
            if top_bids > top_asks * 2:
                return {"type": "liquidity_zone", "side": "bid", "strength": top_bids / max(1.0, top_asks)}
            if top_asks > top_bids * 2:
                return {"type": "liquidity_zone", "side": "ask", "strength": top_asks / max(1.0, top_bids)}
        except Exception:
            return None
        return None

    async def detect_momentum_burst(self, trades: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # detect N consecutive same-side aggressive trades
        if not trades:
            return None
        count = 0
        last_side = None
        for t in trades[:30]:
            side = (t.get("side") or t.get("direction") or "buy").lower()[0]
            if last_side is None:
                last_side = side
                count = 1
            elif side == last_side:
                count += 1
            else:
                break
        if count >= 8:
            return {"type": "momentum_burst", "side": "buy" if last_side == "b" else "sell", "count": count, "confidence": min(1.0, count / 30.0)}
        return None

    async def detect_spoofing_iceberg(self, trades: List[Dict[str, Any]], orderbook: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # very naive: large resting size near top but small executed trades
        if not orderbook or not trades:
            return None
        try:
            bids = orderbook.get("bids") or []
            asks = orderbook.get("asks") or []
            if bids and float(bids[0][1]) > 10_000:  # arbitrary large
                # if executed trades sizes small -> possible iceberg/spoof
                avg_trade = sum(float(t.get("size") or 0) for t in trades[:50]) / max(1, min(50, len(trades)))
                if avg_trade < 1.0:
                    return {"type": "iceberg_hint", "side": "bid", "book_top_size": float(bids[0][1]), "avg_trade": avg_trade}
        except Exception:
            return None
        return None

    async def run_once(self) -> None:
        trades = await self.sample_recent_trades(200)
        ob = await self.sample_orderbook()

        detectors = [
            self.detect_orderflow_imbalance,
            self.detect_volatility_spike,
            self.detect_liquidity_zones,
            self.detect_momentum_burst,
            lambda t, o=ob: self.detect_spoofing_iceberg(trades, ob),
        ]

        events = []
        for d in detectors:
            try:
                if asyncio.iscoroutinefunction(d):
                    res = await d(trades)
                else:
                    res = await d(trades, ob)
                if res:
                    events.append(res)
            except Exception as e:
                logger.debug("Detector error: %s", e)

        # emit events
        for ev in events:
            payload = {
                "symbol": self.symbol,
                "event_type": ev.get("type", "ai_event"),
                "event_data": ev,
            }
            logger.info("AI event: %s", ev)
            if self.writer:
                try:
                    await self.writer.write_market_event(payload)
                except Exception:
                    logger.exception("Failed to write market event")

    async def run(self, interval: float = 10.0) -> None:
        self._running = True
        logger.info("AI Brain (FULL) started for %s", self.symbol)
        try:
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
