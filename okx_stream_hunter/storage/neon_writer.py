import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict

try:
    import asyncpg
except ImportError:
    asyncpg = None

from ..utils.logger import get_logger
from ..config.loader import load_settings

logger = get_logger(__name__)


class NeonWriter:
    """
    Full Production-Grade Neon Database Writer
    """

    def __init__(self) -> None:
        if asyncpg is None:
            raise ImportError("asyncpg is required. Install it first.")

        cfg = load_settings()
        db_cfg = cfg.database

        self.db_url = db_cfg.url
        self.pool_min_size = getattr(db_cfg, "pool_min_size", 1)
        self.pool_max_size = getattr(db_cfg, "pool_max_size", 5)
        self.command_timeout = getattr(db_cfg, "command_timeout", 60)

        self.batch_size = getattr(db_cfg, "batch_size", 50)
        self.batch_timeout = getattr(db_cfg, "batch_timeout", 5)

        self.pool: Optional[asyncpg.Pool] = None
        self._write_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

        self.total_writes = 0
        self.failed_writes = 0
        self.total_batches = 0

    # ========================
    # Timestamp Normalizer
    # ========================
    def _normalize_timestamp(self, ts):
        """Convert ts/OKX timestamp into real datetime always."""
        if ts is None:
            return datetime.now(timezone.utc)

        if isinstance(ts, datetime):
            return ts

        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)

        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except:
                pass

        return datetime.now(timezone.utc)

    # ========================
    # DB Connection
    # ========================
    async def connect(self) -> None:
        logger.info("Connecting to Neon database...")
        self.pool = await asyncpg.create_pool(
            dsn=self.db_url,
            min_size=self.pool_min_size,
            max_size=self.pool_max_size,
            command_timeout=self.command_timeout,
        )
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version();")
            logger.info(f"Connected to Neon DB: {version}")

    async def disconnect(self) -> None:
        if not self.pool:
            return

        await self.flush_all()
        for t in self._flush_tasks.values():
            t.cancel()

        await self.pool.close()
        logger.info("Database connection closed.")

    # ========================
    # Queue Management
    # ========================
    async def _enqueue(self, table: str, item: Dict[str, Any]) -> None:
        async with self._lock:
            self._write_queues[table].append(item)

            if table not in self._flush_tasks:
                self._flush_tasks[table] = asyncio.create_task(self._auto_flush(table))

            if len(self._write_queues[table]) >= self.batch_size:
                asyncio.create_task(self._flush_queue(table))

    # ========================
    # Public write methods
    # ========================
    async def write_trades(self, trades: List[Dict[str, Any]]) -> None:
        for t in trades:
            await self._enqueue("trades", t)

    async def write_orderbook_snapshot(self, snapshot: Dict[str, Any]) -> None:
        await self._enqueue("orderbook_snapshots", snapshot)

    async def write_orderbook_snapshots(self, snapshots: List[Dict[str, Any]]) -> None:
        for s in snapshots:
            await self._enqueue("orderbook_snapshots", s)

    async def write_indicator(self, indicator: Dict[str, Any]) -> None:
        await self._enqueue("indicators", indicator)

    async def write_candle(self, timeframe: str, candle: Dict[str, Any]) -> None:
        await self._enqueue(f"candles_{timeframe}", candle)

    async def write_market_event(self, event: Dict[str, Any]) -> None:
        await self._enqueue("market_events", event)

    async def write_health_metric(self, metric: Dict[str, Any]) -> None:
        await self._enqueue("health_metrics", metric)

    async def write_log(self, log: Dict[str, Any]) -> None:
        await self._enqueue("system_logs", log)

    async def write_data_quality_issue(self, issue: Dict[str, Any]) -> None:
        await self._enqueue("data_quality_logs", issue)

    # ========================
    # Auto Flush
    # ========================
    async def _auto_flush(self, table: str) -> None:
        try:
            while True:
                await asyncio.sleep(self.batch_timeout)
                await self._flush_queue(table)
        except asyncio.CancelledError:
            await self._flush_queue(table)

    # ========================
    # Flushing
    # ========================
    async def _flush_queue(self, table: str) -> None:
        if not self.pool:
            return

        async with self._lock:
            items = self._write_queues[table]
            if not items:
                return

            batch = items[:]
            self._write_queues[table].clear()

        try:
            await self._batch_insert(table, batch)
            self.total_writes += len(batch)
            self.total_batches += 1
        except Exception as e:
            self.failed_writes += len(batch)
            logger.error(f"Failed to flush {table}: {e}", exc_info=True)

    # ========================
    # Batch Insert Logic
    # ========================
    async def _batch_insert(self, table: str, items: List[Dict[str, Any]]) -> None:
        if not items:
            return

        async with self.pool.acquire() as conn:

            # -------------------------
            # CANDLES
            # -------------------------
            if table.startswith("candles_"):
                sql = f"""
                    INSERT INTO {table}
                    (symbol, open_time, close_time, open, high, low, close, volume, trades)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    ON CONFLICT (symbol, open_time) DO UPDATE SET
                        close_time=EXCLUDED.close_time,
                        open=EXCLUDED.open,
                        high=EXCLUDED.high,
                        low=EXCLUDED.low,
                        close=EXCLUDED.close,
                        volume=EXCLUDED.volume,
                        trades=EXCLUDED.trades
                """
                values = [
                    (
                        x.get("symbol"),
                        self._normalize_timestamp(x.get("open_time")),
                        self._normalize_timestamp(x.get("close_time")),
                        x.get("open"),
                        x.get("high"),
                        x.get("low"),
                        x.get("close"),
                        x.get("volume"),
                        x.get("trades", 0),
                    )
                    for x in items
                ]

            # -------------------------
            # INDICATORS
            # -------------------------
            elif table == "indicators":
                sql = """
                    INSERT INTO indicators
                    (symbol,timeframe,timestamp,close,
                     rsi_14,ema_20,ema_50,
                     macd,macd_signal,macd_hist,
                     bb_middle,bb_upper,bb_lower,
                     stoch_rsi,atr_14,volume)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)
                    ON CONFLICT (symbol,timeframe,timestamp) DO UPDATE SET
                        close=EXCLUDED.close,
                        rsi_14=EXCLUDED.rsi_14,
                        ema_20=EXCLUDED.ema_20,
                        ema_50=EXCLUDED.ema_50,
                        macd=EXCLUDED.macd,
                        macd_signal=EXCLUDED.macd_signal,
                        macd_hist=EXCLUDED.macd_hist,
                        bb_middle=EXCLUDED.bb_middle,
                        bb_upper=EXCLUDED.bb_upper,
                        bb_lower=EXCLUDED.bb_lower,
                        stoch_rsi=EXCLUDED.stoch_rsi,
                        atr_14=EXCLUDED.atr_14,
                        volume=EXCLUDED.volume
                """
                values = [
                    (
                        x.get("symbol"),
                        x.get("timeframe"),
                        self._normalize_timestamp(x.get("timestamp")),
                        x.get("close"),
                        x.get("rsi_14"),
                        x.get("ema_20"),
                        x.get("ema_50"),
                        x.get("macd"),
                        x.get("macd_signal"),
                        x.get("macd_hist"),
                        x.get("bb_middle"),
                        x.get("bb_upper"),
                        x.get("bb_lower"),
                        x.get("stoch_rsi"),
                        x.get("atr_14"),
                        x.get("volume"),
                    )
                    for x in items
                ]

            # -------------------------
            # TRADES
            # -------------------------
            elif table == "trades":
                sql = """
                    INSERT INTO trades
                    (symbol, trade_id, side, price, size, timestamp,
                     is_liquidation, raw_json)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    ON CONFLICT (symbol, trade_id) DO NOTHING
                """

                values = []
                for x in items:
                    ts = self._normalize_timestamp(
                        x.get("timestamp") or x.get("ts") or x.get("time") or x.get("T")
                    )

                    trade_id = (
                        x.get("trade_id")
                        or x.get("id")
                        or x.get("tradeId")
                        or x.get("tradeID")
                    )

                    if not trade_id:
                        trade_id = (
                            f"{x.get('symbol')}-"
                            f"{int(ts.timestamp() * 1000)}"
                        )

                    price = x.get("price") or x.get("px")
                    size = x.get("size") or x.get("sz")

                    values.append(
                        (
                            x.get("symbol"),
                            trade_id,
                            x.get("side"),
                            price,
                            size,
                            ts,
                            x.get("is_liquidation", False),
                            json.dumps(x, default=str),
                        )
                    )

            # -------------------------
            # ORDERBOOK SNAPSHOTS
            # -------------------------
            elif table == "orderbook_snapshots":
                sql = """
                    INSERT INTO orderbook_snapshots
                    (symbol,timestamp,best_bid,best_ask,
                     bid_volume,ask_volume,mm_pressure,spread,levels_data)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                """

                values = []
                for x in items:
                    ts = self._normalize_timestamp(
                        x.get("timestamp") or x.get("ts") or x.get("time")
                    )

                    values.append(
                        (
                            x.get("symbol"),
                            ts,
                            x.get("best_bid"),
                            x.get("best_ask"),
                            x.get("bid_volume"),
                            x.get("ask_volume"),
                            x.get("mm_pressure"),
                            x.get("spread"),
                            json.dumps(x.get("levels_data"), default=str),
                        )
                    )

            # -------------------------
            # MARKET EVENTS
            # -------------------------
            elif table == "market_events":
                sql = """
                    INSERT INTO market_events
                    (symbol,event_type,event_data,timestamp)
                    VALUES ($1,$2,$3,$4)
                """
                values = [
                    (
                        x.get("symbol"),
                        x.get("event_type"),
                        json.dumps(x.get("event_data"), default=str),
                        self._normalize_timestamp(x.get("timestamp")),
                    )
                    for x in items
                ]

            # -------------------------
            # HEALTH METRICS
            # -------------------------
            elif table == "health_metrics":
                sql = """
                    INSERT INTO health_metrics
                    (metric_name,metric_value,metric_data,timestamp)
                    VALUES ($1,$2,$3,$4)
                """
                values = [
                    (
                        x.get("metric_name"),
                        x.get("metric_value"),
                        json.dumps(x.get("metric_data"), default=str),
                        self._normalize_timestamp(x.get("timestamp")),
                    )
                    for x in items
                ]

            # -------------------------
            # SYSTEM LOGS
            # -------------------------
            elif table == "system_logs":
                sql = """
                    INSERT INTO system_logs
                    (level,message,context,timestamp)
                    VALUES ($1,$2,$3,$4)
                """
                values = [
                    (
                        x.get("level"),
                        x.get("message"),
                        json.dumps(x.get("context"), default=str),
                        self._normalize_timestamp(x.get("timestamp")),
                    )
                    for x in items
                ]

            # -------------------------
            # DATA QUALITY LOGS
            # -------------------------
            elif table == "data_quality_logs":
                sql = """
                    INSERT INTO data_quality_logs
                    (issue_type,symbol,timeframe,details,timestamp)
                    VALUES ($1,$2,$3,$4,$5)
                """
                values = [
                    (
                        x.get("issue_type"),
                        x.get("symbol"),
                        x.get("timeframe"),
                        json.dumps(x.get("details"), default=str),
                        self._normalize_timestamp(x.get("timestamp")),
                    )
                    for x in items
                ]

            else:
                logger.warning(f"Unknown table {table}, skipping...")
                return

            await conn.executemany(sql, values)

    # ========================
    # Utilities
    # ========================
    async def flush_all(self) -> None:
        for table in list(self._write_queues.keys()):
            await self._flush_queue(table)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_writes": self.total_writes,
            "failed_writes": self.failed_writes,
            "total_batches": self.total_batches,
            "pending": sum(len(x) for x in self._write_queues.values()),
        }


NeonDBWriter = NeonWriter
