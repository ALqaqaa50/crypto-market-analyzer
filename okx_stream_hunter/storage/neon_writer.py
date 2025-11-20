"""
Neon Database Writer - Async batch writer with connection pooling
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import defaultdict

try:
    import asyncpg
except ImportError:
    asyncpg = None

from ..utils.logger import get_logger
from ..config.loader import get_config


logger = get_logger(__name__)


class NeonWriter:
    """
    Async writer for Neon PostgreSQL database.
    Features:
    - Connection pooling
    - Batch inserts
    - Auto-reconnect
    - Error handling
    """
    
    def __init__(self):
        if asyncpg is None:
            raise ImportError("asyncpg is required. Run: pip install asyncpg")
        
        config = get_config()
        
        # Database config
        self.db_url = config.get("database", "url")
        if not self.db_url:
            raise ValueError(
                "Database URL not configured. Set NEON_DATABASE_URL environment variable"
            )
        
        self.pool_min_size = config.get("database", "pool_min_size", default=2)
        self.pool_max_size = config.get("database", "pool_max_size", default=10)
        self.command_timeout = config.get("database", "command_timeout", default=60)
        
        # Batch settings
        self.batch_size = config.get("database", "batch_size", default=100)
        self.batch_timeout = config.get("database", "batch_timeout", default=5)
        
        # State
        self.pool: Optional[asyncpg.Pool] = None
        self._write_queues: Dict[str, List] = defaultdict(list)
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self.total_writes = 0
        self.failed_writes = 0
        self.total_batches = 0
    
    async def connect(self):
        """Initialize database connection pool"""
        if self.pool:
            logger.warning("Pool already initialized")
            return
        
        try:
            logger.info(f"Connecting to Neon database...")
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=self.command_timeout,
            )
            logger.info(
                f"Database pool created (min={self.pool_min_size}, max={self.pool_max_size})"
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Database connected: {version[:50]}...")
        
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self):
        """Close database connection pool"""
        if self.pool:
            # Flush all pending writes
            await self.flush_all()
            
            # Cancel flush tasks
            for task in self._flush_tasks.values():
                task.cancel()
            
            # Close pool
            await self.pool.close()
            self.pool = None
            logger.info("Database pool closed")
    
    async def write_candle(
        self,
        timeframe: str,
        candle: Dict[str, Any]
    ):
        """
        Write a candle to the appropriate timeframe table.
        Uses batch writes for efficiency.
        """
        table_name = f"candles_{timeframe}"
        
        async with self._lock:
            self._write_queues[table_name].append(candle)
            
            # Start flush task if not running
            if table_name not in self._flush_tasks:
                self._flush_tasks[table_name] = asyncio.create_task(
                    self._auto_flush(table_name)
                )
            
            # Immediate flush if batch is full
            if len(self._write_queues[table_name]) >= self.batch_size:
                asyncio.create_task(self._flush_queue(table_name))
    
    async def write_indicator(self, indicator: Dict[str, Any]):
        """Write indicator data"""
        async with self._lock:
            self._write_queues["indicators"].append(indicator)
            
            if "indicators" not in self._flush_tasks:
                self._flush_tasks["indicators"] = asyncio.create_task(
                    self._auto_flush("indicators")
                )
    
    async def write_market_event(
        self,
        symbol: str,
        event_type: str,
        event_data: Dict[str, Any],
        timestamp: datetime
    ):
        """Write market event (liquidation, funding rate, etc.)"""
        event = {
            "symbol": symbol,
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": timestamp,
        }
        
        async with self._lock:
            self._write_queues["market_events"].append(event)
            
            if "market_events" not in self._flush_tasks:
                self._flush_tasks["market_events"] = asyncio.create_task(
                    self._auto_flush("market_events")
                )
    
    async def write_orderbook_snapshot(self, snapshot: Dict[str, Any]):
        """Write order book snapshot"""
        async with self._lock:
            self._write_queues["orderbook_snapshots"].append(snapshot)
            
            if "orderbook_snapshots" not in self._flush_tasks:
                self._flush_tasks["orderbook_snapshots"] = asyncio.create_task(
                    self._auto_flush("orderbook_snapshots")
                )
    
    async def write_health_metric(
        self,
        metric_name: str,
        metric_value: Optional[float],
        metric_data: Optional[Dict],
        timestamp: datetime
    ):
        """Write health metric"""
        metric = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "metric_data": metric_data,
            "timestamp": timestamp,
        }
        
        async with self._lock:
            self._write_queues["health_metrics"].append(metric)
            
            if "health_metrics" not in self._flush_tasks:
                self._flush_tasks["health_metrics"] = asyncio.create_task(
                    self._auto_flush("health_metrics")
                )
    
    async def write_log(
        self,
        level: str,
        message: str,
        context: Optional[Dict],
        timestamp: datetime
    ):
        """Write system log"""
        log = {
            "level": level,
            "message": message,
            "context": context,
            "timestamp": timestamp,
        }
        
        async with self._lock:
            self._write_queues["system_logs"].append(log)
            
            if "system_logs" not in self._flush_tasks:
                self._flush_tasks["system_logs"] = asyncio.create_task(
                    self._auto_flush("system_logs")
                )
    
    async def write_data_quality_issue(
        self,
        issue_type: str,
        symbol: Optional[str],
        timeframe: Optional[str],
        details: Dict,
        timestamp: datetime
    ):
        """Write data quality issue"""
        issue = {
            "issue_type": issue_type,
            "symbol": symbol,
            "timeframe": timeframe,
            "details": details,
            "timestamp": timestamp,
        }
        
        async with self._lock:
            self._write_queues["data_quality_logs"].append(issue)
            
            if "data_quality_logs" not in self._flush_tasks:
                self._flush_tasks["data_quality_logs"] = asyncio.create_task(
                    self._auto_flush("data_quality_logs")
                )
    
    async def _auto_flush(self, table_name: str):
        """Auto-flush queue after timeout"""
        try:
            while True:
                await asyncio.sleep(self.batch_timeout)
                await self._flush_queue(table_name)
        except asyncio.CancelledError:
            # Final flush before exit
            await self._flush_queue(table_name)
    
    async def _flush_queue(self, table_name: str):
        """Flush pending writes for a specific table"""
        if not self.pool:
            logger.warning("Cannot flush: database not connected")
            return
        
        async with self._lock:
            items = self._write_queues[table_name]
            if not items:
                return
            
            # Take items and clear queue
            batch = items[:]
            self._write_queues[table_name].clear()
        
        # Execute batch insert
        try:
            await self._batch_insert(table_name, batch)
            self.total_writes += len(batch)
            self.total_batches += 1
            logger.debug(
                f"Flushed {len(batch)} items to {table_name} "
                f"(total: {self.total_writes})"
            )
        except Exception as e:
            self.failed_writes += len(batch)
            logger.error(f"Failed to flush {table_name}: {e}")
            # Re-queue failed items (optional)
            # async with self._lock:
            #     self._write_queues[table_name].extend(batch)
    
    async def _batch_insert(self, table_name: str, items: List[Dict]):
        """Execute batch insert for specific table"""
        if not items:
            return
        
        # Build SQL based on table
        if table_name.startswith("candles_"):
            sql = f"""
                INSERT INTO {table_name} 
                (symbol, open_time, close_time, open, high, low, close, volume, trades)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (symbol, open_time) DO UPDATE SET
                    close_time = EXCLUDED.close_time,
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    trades = EXCLUDED.trades
            """
            
            values = [
                (
                    item["symbol"],
                    item["open_time"],
                    item["close_time"],
                    item["open"],
                    item["high"],
                    item["low"],
                    item["close"],
                    item["volume"],
                    item.get("trades", 0),
                )
                for item in items
            ]
        
        elif table_name == "indicators":
            sql = """
                INSERT INTO indicators
                (symbol, timeframe, timestamp, close, rsi_14, ema_20, ema_50,
                 macd, macd_signal, macd_hist, bb_middle, bb_upper, bb_lower,
                 stoch_rsi, atr_14, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                ON CONFLICT (symbol, timeframe, timestamp) DO UPDATE SET
                    close = EXCLUDED.close,
                    rsi_14 = EXCLUDED.rsi_14,
                    ema_20 = EXCLUDED.ema_20,
                    ema_50 = EXCLUDED.ema_50,
                    macd = EXCLUDED.macd,
                    macd_signal = EXCLUDED.macd_signal,
                    macd_hist = EXCLUDED.macd_hist,
                    bb_middle = EXCLUDED.bb_middle,
                    bb_upper = EXCLUDED.bb_upper,
                    bb_lower = EXCLUDED.bb_lower,
                    stoch_rsi = EXCLUDED.stoch_rsi,
                    atr_14 = EXCLUDED.atr_14,
                    volume = EXCLUDED.volume
            """
            
            values = [
                (
                    item["symbol"],
                    item["timeframe"],
                    item["timestamp"],
                    item.get("close"),
                    item.get("rsi_14"),
                    item.get("ema_20"),
                    item.get("ema_50"),
                    item.get("macd"),
                    item.get("macd_signal"),
                    item.get("macd_hist"),
                    item.get("bb_middle"),
                    item.get("bb_upper"),
                    item.get("bb_lower"),
                    item.get("stoch_rsi"),
                    item.get("atr_14"),
                    item.get("volume"),
                )
                for item in items
            ]
        
        elif table_name == "market_events":
            sql = """
                INSERT INTO market_events (symbol, event_type, event_data, timestamp)
                VALUES ($1, $2, $3, $4)
            """
            
            values = [
                (
                    item["symbol"],
                    item["event_type"],
                    item["event_data"],
                    item["timestamp"],
                )
                for item in items
            ]
        
        elif table_name == "orderbook_snapshots":
            sql = """
                INSERT INTO orderbook_snapshots
                (symbol, timestamp, best_bid, best_ask, bid_volume, ask_volume,
                 mm_pressure, spread, levels_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """
            
            values = [
                (
                    item["symbol"],
                    item["timestamp"],
                    item.get("best_bid"),
                    item.get("best_ask"),
                    item.get("bid_volume"),
                    item.get("ask_volume"),
                    item.get("mm_pressure"),
                    item.get("spread"),
                    item.get("levels_data"),
                )
                for item in items
            ]
        
        elif table_name == "health_metrics":
            sql = """
                INSERT INTO health_metrics (metric_name, metric_value, metric_data, timestamp)
                VALUES ($1, $2, $3, $4)
            """
            
            values = [
                (
                    item["metric_name"],
                    item.get("metric_value"),
                    item.get("metric_data"),
                    item["timestamp"],
                )
                for item in items
            ]
        
        elif table_name == "system_logs":
            sql = """
                INSERT INTO system_logs (level, message, context, timestamp)
                VALUES ($1, $2, $3, $4)
            """
            
            values = [
                (
                    item["level"],
                    item["message"],
                    item.get("context"),
                    item["timestamp"],
                )
                for item in items
            ]
        
        elif table_name == "data_quality_logs":
            sql = """
                INSERT INTO data_quality_logs
                (issue_type, symbol, timeframe, details, timestamp)
                VALUES ($1, $2, $3, $4, $5)
            """
            
            values = [
                (
                    item["issue_type"],
                    item.get("symbol"),
                    item.get("timeframe"),
                    item["details"],
                    item["timestamp"],
                )
                for item in items
            ]
        
        else:
            logger.error(f"Unknown table: {table_name}")
            return
        
        # Execute batch
        async with self.pool.acquire() as conn:
            await conn.executemany(sql, values)
    
    async def flush_all(self):
        """Flush all pending writes"""
        logger.info("Flushing all pending writes...")
        tables = list(self._write_queues.keys())
        for table in tables:
            await self._flush_queue(table)
        logger.info("All writes flushed")
    
    def get_stats(self) -> Dict:
        """Get writer statistics"""
        pending = sum(len(q) for q in self._write_queues.values())
        
        return {
            "total_writes": self.total_writes,
            "failed_writes": self.failed_writes,
            "total_batches": self.total_batches,
            "pending_writes": pending,
            "queues": {
                table: len(queue)
                for table, queue in self._write_queues.items()
            },
        }