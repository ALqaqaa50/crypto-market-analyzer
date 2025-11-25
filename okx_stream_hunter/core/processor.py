# okx_stream_hunter/core/processor.py

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from okx_stream_hunter.storage.neon_writer import NeonDBWriter
from okx_stream_hunter.modules.whales.detector import WhaleDetector
from okx_stream_hunter.modules.volume.cvd import CVDEngine
from okx_stream_hunter.modules.candles.builder import MultiTimeframeCandleBuilder, Candle


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
    - Feed AI Brain in real-time (if provided)
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        db_writer: Optional[NeonDBWriter] = None,
        db_enable_trades: bool = True,
        db_enable_orderbook: bool = True,
        ai_brain: Optional[Any] = None,
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
        
        # Optional AI Brain for real-time feed
        self.ai_brain = ai_brain
        
        # Whale Detector
        self.whale_detector = WhaleDetector()
        self.whale_events: List[Any] = []
        
        # CVD Engine
        self.cvd_engine = CVDEngine(window_size=1000)
        
        # Track whale detection for dashboard
        self.whale_count = 0
        self.last_whale_event = None
        
        # 🕯️ Candle Builders (1m, 5m, 15m, 1h timeframes)
        self.candle_builders: Dict[str, MultiTimeframeCandleBuilder] = {}
        self.closed_candles: Dict[str, List[Candle]] = {}
        self.candle_timeframes = ["1m", "5m", "15m", "1h"]

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
        
        # 🐋 Whale Detection - check all trades for large orders
        for raw in data:
            try:
                price = float(raw["px"])
                size = float(raw["sz"])
                side = raw.get("side", "").lower()
                ts_ms = float(raw.get("ts", 0.0))
                
                # Check if whale trade
                whale_event = self.whale_detector.check_trade({
                    'price': price,
                    'size': size,
                    'side': side,
                    'timestamp': ts_ms / 1000.0
                })
                
                if whale_event:
                    self.whale_count += 1
                    self.last_whale_event = whale_event
                    self.whale_events.append(whale_event)
                    if len(self.whale_events) > 100:
                        self.whale_events = self.whale_events[-50:]
                    
                    self.logger.warning(
                        f"🐋 WHALE DETECTED! Side={whale_event.side.upper()}, "
                        f"Size={whale_event.size:.2f}, "
                        f"USD=${whale_event.usd_value:,.0f}, "
                        f"Magnitude={whale_event.magnitude:.1f}x"
                    )
                
                # Update CVD Engine
                self.cvd_engine.add_trade({
                    'side': side,
                    'size': size,
                    'price': price,
                    'timestamp': ts_ms / 1000.0
                })
                
                # 🕯️ Build candles from trades
                if inst_id not in self.candle_builders:
                    self.candle_builders[inst_id] = MultiTimeframeCandleBuilder(
                        symbol=inst_id, 
                        timeframes=self.candle_timeframes
                    )
                    self.closed_candles[inst_id] = []
                
                # Process tick and get closed candles
                closed = self.candle_builders[inst_id].process_tick(
                    price=price,
                    size=size,
                    ts=datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                )
                
                if closed:
                    self.closed_candles[inst_id].extend(closed)
                    # Keep only last 1000 candles in memory
                    if len(self.closed_candles[inst_id]) > 1000:
                        self.closed_candles[inst_id] = self.closed_candles[inst_id][-500:]
                    
                    # Update SystemState with candles
                    from okx_stream_hunter.state import get_system_state
                    state = get_system_state()
                    
                    # Organize candles by timeframe
                    candles_by_tf = {}
                    for c in self.closed_candles[inst_id]:
                        if c.timeframe not in candles_by_tf:
                            candles_by_tf[c.timeframe] = []
                        candles_by_tf[c.timeframe].append(c)
                    
                    state.update_candles(
                        candles_1m=candles_by_tf.get("1m", []),
                        candles_5m=candles_by_tf.get("5m", []),
                        candles_15m=candles_by_tf.get("15m", []),
                        candles_1h=candles_by_tf.get("1h", [])
                    )
                    
                    # Optionally save to DB
                    if self.db_writer:
                        for candle in closed:
                            self.logger.info(
                                f"🕯️ Candle closed: {candle.symbol} {candle.timeframe} "
                                f"O={candle.open:.2f} H={candle.high:.2f} "
                                f"L={candle.low:.2f} C={candle.close:.2f} V={candle.volume:.2f}"
                            )
                
            except Exception as e:
                self.logger.debug(f"Whale detection error: {e}")
        
        # Update SystemState with whale events and CVD
        from okx_stream_hunter.state import get_system_state
        state = get_system_state()
        state.update_whale_events(self.whale_events, self.whale_count)
        
        # Get CVD from engine and determine trend
        cvd_data = self.cvd_engine.get_cvd()
        cvd_val = cvd_data.get('cvd', 0.0)
        cvd_trend = "bullish" if cvd_val > 0 else "bearish" if cvd_val < 0 else "neutral"
        state.update_cvd_metrics(cvd_val, cvd_trend)
        
        # Feed AI Brain in real-time (ticker from last trade price)
        if self.ai_brain and bucket:
            last_trade = bucket[-1]
            try:
                self.ai_brain.update_from_ticker({
                    "last": last_trade.price,
                    "ts": int(last_trade.ts * 1000),
                })
                # Also feed trades batch
                self.ai_brain.update_from_trades([
                    {
                        "side": t.side,
                        "size": t.size,
                        "price": t.price,
                        "timestamp": int(t.ts * 1000),
                    }
                    for t in bucket[-50:]  # last 50 trades
                ])
                self.logger.info(
                    f"🔥 AI BRAIN ← TICKER: price={last_trade.price:.2f}, cvd={cvd_value:.2f}, trades={len(bucket[-50:])}"
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to feed AI brain from trades: {e}", exc_info=True)

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
        
        # Feed AI Brain in real-time
        if self.ai_brain and ob:
            try:
                bid_vol_sum = sum(b.size for b in ob.bids[:5]) if ob.bids else 0.0
                ask_vol_sum = sum(a.size for a in ob.asks[:5]) if ob.asks else 0.0
                best_bid = ob.bids[0].price if ob.bids else None
                best_ask = ob.asks[0].price if ob.asks else None
                
                self.ai_brain.update_from_orderbook({
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "bid_volume": bid_vol_sum,
                    "ask_volume": ask_vol_sum,
                    "levels_data": {
                        "bids": {str(b.price): b.size for b in ob.bids[:10]},
                        "asks": {str(a.price): a.size for a in ob.asks[:10]},
                    }
                })
                mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else None
                mid_str = f"{mid:.2f}" if mid else "None"
                self.logger.info(
                    f"🔥 AI BRAIN ← ORDERBOOK: bid={best_bid}, ask={best_ask}, mid={mid_str}, "
                    f"bid_vol={bid_vol_sum:.2f}, ask_vol={ask_vol_sum:.2f}"
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to feed AI brain from orderbook: {e}", exc_info=True)

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
