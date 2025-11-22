# okx_stream_hunter/core/processor.py

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from okx_stream_hunter.storage.neon_writer import NeonDBWriter


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

    Responsibilities:
    - Normalize trades and orderbook messages
    - Maintain light-weight in-memory state
    - Compute simple CVD and mid-price
    - Optionally persist data to PostgreSQL via NeonDBWriter
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        db_writer: Optional[NeonDBWriter] = None,
        db_enable_trades: bool = True,
        db_enable_orderbook: bool = True,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)

        # Latest trades by instrument
        self.trades: Dict[str, List[Trade]] = {}

        # Latest orderbook snapshot by instrument
        self.orderbooks: Dict[str, OrderBookSnapshot] = {}

        # Simple cumulative volume delta per instrument
        self.cvd: Dict[str, float] = {}

        # Last processed timestamp (epoch seconds)
        self.last_update_ts: float = 0.0

        # Optional DB writer
        self.db_writer = db_writer
        self.db_enable_trades = db_enable_trades
        self.db_enable_orderbook = db_enable_orderbook

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
            self.logger.debug(f"Ignoring unsupported channel {channel!r}")

    # ------------------------------------------------------------------
    async def _handle_trades(self, inst_id: str, msg: Dict[str, Any]) -> None:
        data = msg.get("data") or []
        if not data:
            return

        bucket = self.trades.setdefault(inst_id, [])
        cvd_value = self.cvd.get(inst_id, 0.0)

        db_rows: List[Dict[str, Any]] = []

        for raw in data:
            try:
                price = float(raw["px"])
                size = float(raw["sz"])
                side = "buy" if raw.get("side") == "buy" else "sell"
                ts_ms = float(raw.get("ts", 0.0))
                ts_sec = ts_ms / 1000.0
            except Exception:
                self.logger.debug(f"Invalid trade payload: {raw}")
                continue

            trade = Trade(ts=ts_sec, price=price, size=size, side=side)
            bucket.append(trade)

            if side == "buy":
                cvd_value += size
            else:
                cvd_value -= size

            if self.db_writer and self.db_enable_trades:
                dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
                db_rows.append(
                    {
                        "symbol": inst_id,
                        "ts": dt,
                        "side": side,
                        "price": price,
                        "size_base": size,
                        "value_quote": price * size,
                        "source": "live",
                    }
                )

        # keep last N trades only (for memory)
        if len(bucket) > 2000:
            del bucket[:-1000]

        self.cvd[inst_id] = cvd_value

        if db_rows and self.db_writer and self.db_enable_trades:
            await self.db_writer.write_trades(db_rows)

    # ------------------------------------------------------------------
    async def _handle_orderbook(self, inst_id: str, msg: Dict[str, Any]) -> None:
        data = msg.get("data") or []
        if not data:
            return

        snapshot_raw = data[0]
        try:
            ts_ms = float(snapshot_raw.get("ts", 0.0))
        except Exception:
            ts_ms = 0.0
        ts_sec = ts_ms / 1000.0

        bids = [
            OrderBookLevel(price=float(px), size=float(sz))
            for px, sz, *_ in snapshot_raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(px), size=float(sz))
            for px, sz, *_ in snapshot_raw.get("asks", [])
        ]

        ob = OrderBookSnapshot(ts=ts_sec, bids=bids, asks=asks)
        self.orderbooks[inst_id] = ob

        if self.db_writer and self.db_enable_orderbook:
            best_bid_px = bids[0].price if bids else None
            best_bid_sz = bids[0].size if bids else None
            best_ask_px = asks[0].price if asks else None
            best_ask_sz = asks[0].size if asks else None

            mid_price: Optional[float] = None
            spread: Optional[float] = None
            if best_bid_px is not None and best_ask_px is not None:
                mid_price = (best_bid_px + best_ask_px) / 2.0
                spread = best_ask_px - best_bid_px

            dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)

            snapshot_row: Dict[str, Any] = {
                "symbol": inst_id,
                "ts": dt,
                "best_bid_px": best_bid_px,
                "best_bid_sz": best_bid_sz,
                "best_ask_px": best_ask_px,
                "best_ask_sz": best_ask_sz,
                "mid_price": mid_price,
                "spread": spread,
                "bid_liq_near": None,
                "ask_liq_near": None,
                "bid_liq_far": None,
                "ask_liq_far": None,
                "raw_bids": snapshot_raw.get("bids"),
                "raw_asks": snapshot_raw.get("asks"),
            }

            await self.db_writer.write_orderbook_snapshots([snapshot_row])

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
