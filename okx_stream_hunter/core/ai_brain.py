# okx_stream_hunter/core/ai_brain.py

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, Optional, TypedDict


Direction = Literal["long", "short", "flat"]


class Signal(TypedDict, total=False):
    direction: Direction
    confidence: float
    reason: str
    price: Optional[float]
    timestamp: float
    regime: str
    scores: Dict[str, float]


@dataclass
class BrainConfig:
    # windows
    price_window: int = 120          # ~ last 2 minutes if 1s ticks
    orderflow_window: int = 60       # trades aggregation window
    volatility_window: int = 60

    # thresholds
    min_trades_volume: float = 0.5   # ignore micro noise
    min_orderbook_volume: float = 5.0
    min_confidence: float = 0.15

    ema_fast_span: int = 12
    ema_slow_span: int = 48

    high_volatility_thr: float = 0.003   # 0.3% std of returns
    trend_strength_thr: float = 0.002    # 0.2% ema gap

    spoof_ratio_thr: float = 3.0         # large walls vs average
    spread_widen_thr: float = 0.0015     # 0.15%

    max_age_seconds: int = 10            # if no fresh data → flat


@dataclass
class MarketState:
    last_price: Optional[float] = None
    last_ts: Optional[float] = None

    prices: Deque[float] = field(default_factory=deque)
    timestamps: Deque[float] = field(default_factory=deque)

    # EMA
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None

    # order flow
    buy_volume_window: Deque[float] = field(default_factory=deque)
    sell_volume_window: Deque[float] = field(default_factory=deque)
    cvd: float = 0.0  # Cumulative Volume Delta

    # orderbook
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    spread: Optional[float] = None

    last_book_levels: Optional[Dict[str, Dict[str, float]]] = None  # for spoofing check


class AIBrain:
    """
    Advanced AI Brain (Phase 1 – Analysis Only).

    Responsibilities:
      - Track short/medium price structure (EMAs, volatility).
      - Measure order flow imbalance from trades.
      - Measure orderbook imbalance and spread/walls.
      - Detect simple spoofing / liquidity games.
      - Classify regime (trend up / trend down / range / chaos).
      - Emit trading signal: direction + confidence + explanation.
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[BrainConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.symbol = symbol
        self.config = config or BrainConfig()
        self.state = MarketState()

        self.log = logger or logging.getLogger("ai_brain")
        self._last_signal: Optional[Signal] = None

    # ------------------------------------------------------------------
    # Public API – called by stream manager / AI loop
    # ------------------------------------------------------------------
    def update_from_ticker(self, ticker: Dict) -> None:
        """
        ticker example (OKX):
        {
            "instId": "BTC-USDT-SWAP",
            "last": "102345.5",
            "ts": "1732350000000",
            ...
        }
        """
        try:
            price = float(ticker.get("last") or ticker.get("px"))
        except (TypeError, ValueError):
            return

        ts_raw = ticker.get("ts") or ticker.get("timestamp") or time.time() * 1000
        try:
            ts = float(ts_raw) / 1000.0
        except (TypeError, ValueError):
            ts = time.time()

        self._update_price_series(price, ts)
        ema_fast_str = f"{self.state.ema_fast:.2f}" if self.state.ema_fast else "None"
        ema_slow_str = f"{self.state.ema_slow:.2f}" if self.state.ema_slow else "None"
        self.log.info(f"✅ AI Brain received TICKER: price={price:.2f}, ema_fast={ema_fast_str}, ema_slow={ema_slow_str}")

    def update_from_trades(self, trades: List[Dict]) -> None:
        """
        trades: list of trade objects from OKX stream or DB.
        We only aggregate buy/sell volume per window.
        """
        if not trades:
            return

        buy_vol = 0.0
        sell_vol = 0.0

        for t in trades:
            side = (t.get("side") or t.get("sd") or "").lower()
            size_raw = t.get("sz") or t.get("size") or t.get("qty") or 0
            try:
                size = abs(float(size_raw))
            except (TypeError, ValueError):
                continue

            if side == "buy":
                buy_vol += size
            elif side == "sell":
                sell_vol += size

        if buy_vol == 0 and sell_vol == 0:
            return

        self._update_orderflow(buy_vol, sell_vol)
        
        # Update CVD (Cumulative Volume Delta)
        self.state.cvd += (buy_vol - sell_vol)
        
        total_buy = sum(self.state.buy_volume_window)
        total_sell = sum(self.state.sell_volume_window)
        self.log.info(f"✅ AI Brain received TRADES: {len(trades)} trades | buy_vol={buy_vol:.2f}, sell_vol={sell_vol:.2f} | total_buy={total_buy:.2f}, total_sell={total_sell:.2f}")

    def update_from_orderbook(self, snapshot: Dict) -> None:
        """
        snapshot: preprocessed orderbook info.
        Expect keys: best_bid, best_ask, bid_volume, ask_volume, levels_data (optional).
        """
        try:
            best_bid = float(snapshot.get("best_bid")) if snapshot.get("best_bid") else None
            best_ask = float(snapshot.get("best_ask")) if snapshot.get("best_ask") else None
        except (TypeError, ValueError):
            best_bid = best_ask = None

        bid_vol = snapshot.get("bid_volume")
        ask_vol = snapshot.get("ask_volume")

        try:
            bid_vol = float(bid_vol) if bid_vol is not None else None
            ask_vol = float(ask_vol) if ask_vol is not None else None
        except (TypeError, ValueError):
            bid_vol = ask_vol = None

        st = self.state
        st.best_bid = best_bid
        st.best_ask = best_ask
        st.bid_volume = bid_vol
        st.ask_volume = ask_vol

        if best_bid and best_ask and best_ask > 0:
            st.spread = (best_ask - best_bid) / best_ask
        else:
            st.spread = None

        levels_data = snapshot.get("levels_data")
        if isinstance(levels_data, dict):
            self._detect_spoofing(levels_data)
        
        mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else None
        mid_str = f"{mid:.2f}" if mid else "None"
        bid_vol_str = f"{bid_vol:.2f}" if bid_vol else "None"
        ask_vol_str = f"{ask_vol:.2f}" if ask_vol else "None"
        spread_str = f"{st.spread:.4f}" if st.spread else "None"
        self.log.info(
            f"✅ AI Brain received ORDERBOOK: bid={best_bid}, ask={best_ask}, mid={mid_str}, "
            f"bid_vol={bid_vol_str}, ask_vol={ask_vol_str}, spread={spread_str}"
        )

    def update_from_indicator(self, indicator: Dict) -> None:
        """
        Optional: attach precomputed indicators (RSI/MACD...) if needed.
        For now, just store them; can be used later to boost scores.
        """
        self.state.indicator = indicator  # type: ignore[attr-defined]

    def build_signal(self) -> Signal:
        """
        Build trading signal based on current state.
        """
        now = time.time()
        st = self.state
        cfg = self.config

        # No price yet
        if not st.last_price or not st.last_ts:
            sig: Signal = {
                "direction": "flat",
                "confidence": 0.0,
                "reason": "no_price",
                "price": None,
                "timestamp": now,
                "regime": "unknown",
                "scores": {},
            }
            self._last_signal = sig
            return sig

        # Stale market
        if now - st.last_ts > cfg.max_age_seconds:
            sig = {
                "direction": "flat",
                "confidence": 0.0,
                "reason": "stale_market",
                "price": st.last_price,
                "timestamp": now,
                "regime": "unknown",
                "scores": {},
            }
            self._last_signal = sig
            return sig

        vol = self._estimate_volatility()
        trend_score, regime = self._trend_score(vol)
        of_score = self._orderflow_score()
        ob_score, _ = self._orderbook_score()
        spoof_score = self._spoofing_score()

        long_score = 0.0
        short_score = 0.0
        neutral_score = 0.0

        # Trend
        if trend_score > 0:
            long_score += trend_score
        elif trend_score < 0:
            short_score += -trend_score
        else:
            neutral_score += 0.2

        # Order flow
        if of_score > 0:
            long_score += 0.8 * of_score
        elif of_score < 0:
            short_score += -0.8 * of_score

        # Orderbook imbalance
        if ob_score > 0:
            long_score += 0.6 * ob_score
        elif ob_score < 0:
            short_score += -0.6 * ob_score

        # Risk penalties
        risk_penalty = max(0.0, spoof_score)
        if vol is not None and vol > cfg.high_volatility_thr * 3:
            risk_penalty += 0.3

        # Direction decision
        if long_score > short_score and long_score > neutral_score:
            direction: Direction = "long"
            raw_conf = long_score - risk_penalty
        elif short_score > long_score and short_score > neutral_score:
            direction = "short"
            raw_conf = short_score - risk_penalty
        else:
            direction = "flat"
            raw_conf = neutral_score - risk_penalty

        confidence = self._score_to_confidence(raw_conf)

        reason_parts: List[str] = []
        if regime != "range":
            reason_parts.append(f"regime={regime}")
        if abs(of_score) > 0.15:
            reason_parts.append(f"orderflow={'buy' if of_score > 0 else 'sell'}")
        if abs(ob_score) > 0.15:
            reason_parts.append(f"orderbook={'bid' if ob_score > 0 else 'ask'}")
        if spoof_score > 0.1:
            reason_parts.append("spoof_risk")
        if vol is not None:
            reason_parts.append(f"vol={vol:.4f}")

        reason = ",".join(reason_parts) or "neutral"

        scores = {
            "trend": trend_score,
            "orderflow": of_score,
            "orderbook": ob_score,
            "spoof": spoof_score,
            "risk_penalty": risk_penalty,
        }

        sig: Signal = {
            "direction": direction,
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason,
            "price": st.last_price,
            "timestamp": now,
            "regime": regime,
            "scores": scores,
        }

        if sig["confidence"] < cfg.min_confidence:
            sig["direction"] = "flat"
            sig["reason"] = "low_confidence;" + reason

        self._last_signal = sig
        return sig

    def get_last_signal(self) -> Optional[Signal]:
        return self._last_signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_price_series(self, price: float, ts: float) -> None:
        st = self.state
        cfg = self.config

        st.last_price = price
        st.last_ts = ts

        st.prices.append(price)
        st.timestamps.append(ts)

        if len(st.prices) > cfg.price_window:
            st.prices.popleft()
            st.timestamps.popleft()

        st.ema_fast = self._ema_update(st.ema_fast, price, cfg.ema_fast_span)
        st.ema_slow = self._ema_update(st.ema_slow, price, cfg.ema_slow_span)

    def _ema_update(self, prev: Optional[float], price: float, span: int) -> float:
        if prev is None:
            return price
        alpha = 2 / (span + 1)
        return prev + alpha * (price - prev)

    def _update_orderflow(self, buy_vol: float, sell_vol: float) -> None:
        st = self.state
        cfg = self.config

        st.buy_volume_window.append(buy_vol)
        st.sell_volume_window.append(sell_vol)

        if len(st.buy_volume_window) > cfg.orderflow_window:
            st.buy_volume_window.popleft()
            st.sell_volume_window.popleft()

    def _estimate_volatility(self) -> Optional[float]:
        st = self.state
        cfg = self.config

        if len(st.prices) < 4:
            return None

        returns: List[float] = []
        for i in range(1, len(st.prices)):
            p0 = st.prices[i - 1]
            p1 = st.prices[i]
            if p0 <= 0:
                continue
            returns.append((p1 - p0) / p0)

        if len(returns) < 3:
            return None

        window = returns[-cfg.volatility_window :]
        mean = sum(window) / len(window)
        var = sum((r - mean) ** 2 for r in window) / max(1, len(window) - 1)
        return math.sqrt(var)

    def _trend_score(self, volatility: Optional[float]) -> (float, str):
        st = self.state
        cfg = self.config

        if st.ema_fast is None or st.ema_slow is None:
            return 0.0, "unknown"

        gap = (st.ema_fast - st.ema_slow) / st.ema_slow

        if abs(gap) < cfg.trend_strength_thr:
            return 0.0, "range"

        if volatility is not None and volatility < cfg.high_volatility_thr:
            vol_boost = 1.2
        else:
            vol_boost = 1.0

        if gap > 0:
            regime = "trend_up"
            return float(min(1.5, gap / cfg.trend_strength_thr)) * vol_boost, regime
        else:
            regime = "trend_down"
            return float(-min(1.5, gap / cfg.trend_strength_thr)) * vol_boost, regime

    def _orderflow_score(self) -> float:
        st = self.state
        cfg = self.config

        total_buy = sum(st.buy_volume_window)
        total_sell = sum(st.sell_volume_window)
        total = total_buy + total_sell

        if total < cfg.min_trades_volume:
            return 0.0

        imbalance = (total_buy - total_sell) / total
        imbalance = max(-1.0, min(1.0, imbalance))
        return float(imbalance)

    def _orderbook_score(self) -> (float, Dict[str, float]):
        st = self.state
        cfg = self.config

        if st.bid_volume is None or st.ask_volume is None:
            return 0.0, {}

        total = st.bid_volume + st.ask_volume
        if total < cfg.min_orderbook_volume:
            return 0.0, {}

        imbalance = (st.bid_volume - st.ask_volume) / total
        imbalance = max(-1.0, min(1.0, imbalance))

        info = {
            "bid_volume": st.bid_volume,
            "ask_volume": st.ask_volume,
            "spread": st.spread or 0.0,
        }

        if st.spread and st.spread > cfg.spread_widen_thr:
            imbalance *= 0.8

        return float(imbalance), info

    def _detect_spoofing(self, levels: Dict[str, Dict[str, float]]) -> None:
        self.state.last_book_levels = levels

    def _spoofing_score(self) -> float:
        st = self.state
        cfg = self.config

        levels = st.last_book_levels
        if not levels or not st.last_price:
            return 0.0

        bids_raw = levels.get("bids", {})
        asks_raw = levels.get("asks", {})

        bids = {float(k): float(v) for k, v in bids_raw.items()}
        asks = {float(k): float(v) for k, v in asks_raw.items()}

        all_vols: List[float] = list(bids.values()) + list(asks.values())
        if len(all_vols) < 4:
            return 0.0

        avg = sum(all_vols) / len(all_vols)
        if avg <= 0:
            return 0.0

        max_bid_price, max_bid_vol = (0.0, 0.0)
        if bids:
            max_bid_price, max_bid_vol = max(bids.items(), key=lambda x: x[1])

        min_ask_price, max_ask_vol = (0.0, 0.0)
        if asks:
            min_ask_price, max_ask_vol = max(asks.items(), key=lambda x: x[1])

        spoof_score = 0.0

        if max_bid_vol / avg > cfg.spoof_ratio_thr and st.last_price > 0:
            dist = abs(st.last_price - max_bid_price) / st.last_price
            spoof_score += max(0.0, 1.0 - dist * 50)

        if max_ask_vol / avg > cfg.spoof_ratio_thr and st.last_price > 0:
            dist = abs(st.last_price - min_ask_price) / st.last_price
            spoof_score += max(0.0, 1.0 - dist * 50)

        return float(max(0.0, min(1.5, spoof_score)))

    def _score_to_confidence(self, raw: float) -> float:
        return 1.0 / (1.0 + math.exp(-raw * 2.0))
