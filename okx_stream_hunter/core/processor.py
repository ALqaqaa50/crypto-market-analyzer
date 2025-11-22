# okx_stream_hunter/core/processor.py

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Trade:
    ts: float
    price: float
    size: float
    side: str  # "buy" or "sell"


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBookSnapshot:
    ts: float
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)


class MarketProcessor:
    """
    Market data processor for OKX streams.

    - Normalizes trades/orderbook messages
    - Maintains light-weight in-memory state
    - Computes simple volume metrics (CVD, VWAP skeleton)
    - Prepares data for storage / further AI / backtesting layers
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

        # Latest trades by instrument
        self.trades: Dict[str, List[Trade]] = {}

        # Latest orderbook snapshot by instrument
        self.orderbooks: Dict[str, OrderBookSnapshot] = {}

        # Simple cumulative volume delta per instrument
        self.cvd: Dict[str, float] = {}

        # Last processed timestamp
        self.last_update_ts: float = 0.0

    # ------------------------------------------------------------------
    async def handle_message(self, msg: Dict[str, Any]) -> None:
        """
        Entry point called by WebSocket client.

        OKX WS payload examples:
        - trades: {"arg": {...}, "data": [{...}, {...}]}
        - books5: {"arg": {...}, "data": [{...}]}
        """
        arg = msg.get("arg", {})
        channel = arg.get("channel")
        inst_id = arg.get("instId")

        if not channel or not inst_id:
            # Could be system / status message
            return

        self.last_update_ts = time.time()

        if channel.startswith("trades"):
            await self._handle_trades(inst_id, msg)
        elif channel.startswith("books"):
            await self._handle_orderbook(inst_id, msg)
        else:
            # other channels can be handled later (tickers, funding, liquidations, ...)
            self.logger.debug(f"Ignoring channel {channel}")

    # ------------------------------------------------------------------
    async def _handle_trades(self, inst_id: str, msg: Dict[str, Any]) -> None:
        data = msg.get("data") or []
        if not data:
            return

        bucket = self.trades.setdefault(inst_id, [])
        cvd_value = self.cvd.get(inst_id, 0.0)

        for raw in data:
            try:
                price = float(raw["px"])
                size = float(raw["sz"])
                side = "buy" if raw.get("side") == "buy" else "sell"
                ts = float(raw.get("ts", 0)) / 1000.0
            except Exception:
                self.logger.debug(f"Invalid trade payload: {raw}")
                continue

            trade = Trade(ts=ts, price=price, size=size, side=side)
            bucket.append(trade)

            if side == "buy":
                cvd_value += size
            else:
                cvd_value -= size

        # keep last N trades only (for memory)
        if len(bucket) > 2000:
            del bucket[:-1000]

        self.cvd[inst_id] = cvd_value

    # ------------------------------------------------------------------
    async def _handle_orderbook(self, inst_id: str, msg: Dict[str, Any]) -> None:
        data = msg.get("data") or []
        if not data:
            return

        snapshot = data[0]
        ts = float(snapshot.get("ts", 0)) / 1000.0

        bids = [
            OrderBookLevel(price=float(px), size=float(sz))
            for px, sz, *_ in snapshot.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(px), size=float(sz))
            for px, sz, *_ in snapshot.get("asks", [])
        ]

        ob = OrderBookSnapshot(ts=ts, bids=bids, asks=asks)
        self.orderbooks[inst_id] = ob

    # ------------------------------------------------------------------
    # Simple analytics helpers
    # ------------------------------------------------------------------
    def get_cvd(self, inst_id: str) -> float:
        return self.cvd.get(inst_id, 0.0)

    def best_bid_ask(self, inst_id: str) -> Optional[Tuple[float, float]]:
        ob = self.orderbooks.get(inst_id)
        if not ob or not ob.bids or not ob.asks:
            return None
        best_bid = ob.bids[0].price
        best_ask = ob.asks[0].price
        return best_bid, best_ask

    def mid_price(self, inst_id: str) -> Optional[float]:
        ba = self.best_bid_ask(inst_id)
        if not ba:
            return None
        bid, ask = ba
        return (bid + ask) / 2.0
