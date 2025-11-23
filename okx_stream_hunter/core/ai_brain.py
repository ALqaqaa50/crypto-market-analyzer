import asyncio
import logging
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except Exception:  # pragma: no cover
    asyncpg = None

logger = logging.getLogger("core.ai_brain")


class CoreAIBrain:
    """
    Full AI Brain for core package.

    - Reads recent trades, latest orderbook snapshot and recent ticker/candle data (when available)
    - Runs a set of lightweight heuristic detectors
    - Produces a single high-level signal: LONG / SHORT / FLAT
      with `confidence` (0.0-1.0) and `reasons` list explaining the decision

    Designed to be safe (no direct order execution), emits events via
    a provided `writer` (must implement `write_market_event(dict)` async).
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool], writer: Any = None, symbol: str = "BTC-USDT-SWAP"):
        self.db_pool = db_pool
        self.writer = writer
        self.symbol = symbol
        self._stop = False

    # -------------------------
    # Data helpers
    # -------------------------
    async def _fetch_trades(self, limit: int = 300) -> List[Dict[str, Any]]:
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
            logger.debug("Failed to fetch trades", exc_info=True)
            return []

    async def _fetch_orderbook(self) -> Optional[Dict[str, Any]]:
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
            logger.debug("Failed to fetch orderbook", exc_info=True)
            return None

    async def _fetch_ticker(self) -> Optional[Dict[str, Any]]:
        # try to read latest ticker/candle — many projects store under `tickers` or `candles_*`
        if self.db_pool is None:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT last, bid, ask, ts FROM tickers WHERE symbol=$1 ORDER BY ts DESC LIMIT 1;",
                    self.symbol,
                )
                if row:
                    return dict(row)
        except Exception:
            # ignore missing table
            pass
        return None

    # -------------------------
    # Detectors (heuristics)
    # -------------------------
    def _orderflow_score(self, trades: List[Dict[str, Any]]) -> float:
        # returns score in [-1,1] where positive = buy pressure
        buy = 0.0
        sell = 0.0
        for t in trades:
            side = (t.get("side") or t.get("direction") or "buy").lower()
            size = float(t.get("size") or t.get("qty") or 0)
            if side.startswith("b"):
                buy += size
            else:
                sell += size
        total = buy + sell
        if total <= 0:
            return 0.0
        return (buy - sell) / total

    def _momentum_score(self, trades: List[Dict[str, Any]]) -> float:
        # compute short-term price momentum: last price - price N ago
        prices = [float(t.get("price") or t.get("last") or 0) for t in trades if t.get("price") or t.get("last")]
        if len(prices) < 5:
            return 0.0
        # prices sorted newest-first from query; invert to chronological
        prices = list(reversed(prices))
        recent = prices[-1]
        past = prices[-6] if len(prices) > 6 else prices[0]
        if past <= 0:
            return 0.0
        return (recent - past) / past

    def _liquidity_pressure(self, orderbook: Optional[Dict[str, Any]]) -> float:
        # positive = bid-side deeper, negative = ask-side deeper
        if not orderbook:
            return 0.0
        bids = orderbook.get("bids") or []
        asks = orderbook.get("asks") or []
        try:
            bid_vol = sum(float(b[1]) for b in bids[:10]) if bids else 0.0
            ask_vol = sum(float(a[1]) for a in asks[:10]) if asks else 0.0
            tot = bid_vol + ask_vol
            if tot <= 0:
                return 0.0
            return (bid_vol - ask_vol) / tot
        except Exception:
            return 0.0

    def _volatility_indicator(self, trades: List[Dict[str, Any]]) -> float:
        # simple heuristic: standard deviation relative to mean
        try:
            prices = [float(t.get("price") or t.get("last") or 0) for t in trades if t.get("price") or t.get("last")]
            if len(prices) < 8:
                return 0.0
            import statistics

            stdev = statistics.pstdev(prices[-50:]) if len(prices) >= 2 else 0.0
            mean = statistics.mean(prices[-50:]) if prices else 0.0
            if mean <= 0:
                return 0.0
            return stdev / mean
        except Exception:
            return 0.0

    # -------------------------
    # Decision logic
    # -------------------------
    def _aggregate_signal(self, of_score: float, mom: float, liq: float, vol: float) -> Dict[str, Any]:
        # weighted combination → raw score in [-1,1]
        # weights chosen heuristically (tweakable)
        w_of = 0.45
        w_mom = 0.25
        w_liq = 0.2
        w_vol = 0.1
        raw = w_of * of_score + w_mom * (mom * 2.0) + w_liq * liq - w_vol * vol

        # normalize and produce confidence
        import math

        confidence = min(1.0, abs(raw) + min(0.5, abs(mom)))
        if abs(raw) < 0.06:
            signal = "FLAT"
        elif raw > 0:
            signal = "LONG"
        else:
            signal = "SHORT"

        reasons = []
        if abs(of_score) > 0.15:
            reasons.append(f"orderflow_imbalance={of_score:.2f}")
        if abs(mom) > 0.005:
            reasons.append(f"momentum={mom:.3f}")
        if abs(liq) > 0.1:
            reasons.append(f"liquidity_pressure={liq:.2f}")
        if vol > 0.002:
            reasons.append(f"volatility={vol:.4f}")

        return {"signal": signal, "confidence": float(confidence), "raw_score": float(raw), "reasons": reasons}

    # -------------------------
    # Run loop
    # -------------------------
    async def run_once(self) -> Dict[str, Any]:
        trades = await self._fetch_trades(300)
        ob = await self._fetch_orderbook()
        ticker = await self._fetch_ticker()

        of_score = self._orderflow_score(trades)
        mom = self._momentum_score(trades)
        liq = self._liquidity_pressure(ob)
        vol = self._volatility_indicator(trades)

        decision = self._aggregate_signal(of_score, mom, liq, vol)

        event = {
            "symbol": self.symbol,
            "event_type": "AI_SIGNAL",
            "event_data": {
                "decision": decision,
                "metrics": {"orderflow": of_score, "momentum": mom, "liquidity": liq, "volatility": vol},
                "ticker": ticker,
            },
        }

        # write to writer if available (best-effort)
        if self.writer is not None:
            try:
                await self.writer.write_market_event(event)
            except Exception:
                logger.exception("Failed to write AI signal to writer")

        logger.info("AI signal %s confidence=%.2f reasons=%s", decision["signal"], decision["confidence"], decision["reasons"])
        return event

    async def run(self, interval: float = 5.0) -> None:
        logger.info("Core AI Brain starting for %s", self.symbol)
        self._stop = False
        try:
            while not self._stop:
                try:
                    await self.run_once()
                except Exception:
                    logger.exception("Error during AI run_once")
                await asyncio.sleep(interval)
        finally:
            logger.info("Core AI Brain stopped")

    def stop(self) -> None:
        self._stop = True


__all__ = ["CoreAIBrain"]
