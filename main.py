# main.py
# ============================================================
#  👑 CRYPTO MARKET ANALYZER - BEAST MAIN LAUNCHER
# ============================================================
#  - Loads settings from config/loader.py
#  - Connects to Neon (Postgres) using NeonDBWriter
#  - Starts OKX StreamEngine (WS + REST ingestion)
#  - Runs background tasks:
#       * health_monitor      → DB + stream health
#       * ai_brain_loop       → placeholder AI brain
#       * dashboard_server    → FastAPI dashboard (optional)
#  - Single entrypoint:
#       python main.py
# ============================================================

import asyncio
import logging
import os
from typing import List, Optional

try:
    import asyncpg  # type: ignore
except Exception:  # pragma: no cover
    asyncpg = None

# --- Local imports from the package ---
from okx_stream_hunter.config.loader import load_settings
from okx_stream_hunter.core import StreamEngine
from okx_stream_hunter.storage.neon_writer import NeonDBWriter


# ============================================================
#  Helpers
# ============================================================

def setup_logging() -> None:
    """
    Configure root logging in a production-friendly format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # Reduce noisy loggers if needed
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def create_db_writer_if_enabled(settings) -> Optional[NeonDBWriter]:
    """
    Create NeonDBWriter only if database is enabled in settings.
    """
    logger = logging.getLogger("db-init")
    db_cfg = settings.database

    if not getattr(db_cfg, "enabled", False):
        logger.info("Database is disabled in settings → running without DB.")
        return None

    # Allow override via env NEON_DATABASE_URL
    env_url = os.getenv("NEON_DATABASE_URL", "").strip()
    if env_url:
        db_cfg.url = env_url

    if not getattr(db_cfg, "url", "").strip():
        logger.warning(
            "Database is enabled but URL is empty. "
            "Set NEON_DATABASE_URL or database.url in settings."
        )
        return None

    writer_logger = logging.getLogger("neon-writer")
    writer = NeonDBWriter()
    # NeonDBWriter internally reads get_settings(), لكن لا مشكلة:
    # هو يستخدم نفس settings / env.
    await writer.connect()
    writer_logger.info("NeonDBWriter connected and ready.")
    return writer


# ============================================================
#  Background Tasks
# ============================================================

async def health_monitor_task(
    db_pool: Optional[asyncpg.Pool],
    interval_sec: int = 60,
) -> None:
    """
    Simple health monitor:
      - Checks DB connectivity
      - Optionally logs counts from key tables
    """
    logger = logging.getLogger("health-monitor")

    if db_pool is None:
        logger.info("DB pool is None → health monitor will only log heartbeat.")
    else:
        logger.info("Health monitor started with DB checks.")

    while True:
        try:
            if db_pool is not None:
                async with db_pool.acquire() as conn:
                    # These SELECTs are safe حتى لو الجداول فاضية.
                    counts = {}
                    for table in [
                        "trades",
                        "orderbook_snapshots",
                        "indicators",
                        "candles_1m",
                    ]:
                        try:
                            q = f"SELECT COUNT(*) FROM {table};"
                            counts[table] = await conn.fetchval(q)
                        except Exception:
                            # جدول غير موجود؟ عادي، نتجاهله
                            counts[table] = None

                    logger.info(f"DB Health: {counts}")
            else:
                logger.info("Health: DB disabled, heartbeat OK.")

        except Exception as e:
            logger.warning(f"Health monitor error: {e}", exc_info=True)

        await asyncio.sleep(interval_sec)


async def ai_brain_loop(
    db_pool: Optional[asyncpg.Pool],
    writer: Optional[NeonDBWriter],
    interval_sec: int = 30,
) -> None:
    """
    Placeholder AI brain:
      - Reads last price / candle if possible
      - Produces a simple pseudo-signal
      - Writes 'market_events' into Neon via NeonDBWriter

    لاحقاً نستبدل هذا بالكامل بنموذج CNN / XGBoost الحقيقي.
    """
    logger = logging.getLogger("ai-brain")

    if db_pool is None or writer is None:
        logger.info("AI Brain running in LIMITED mode (no DB or writer).")

    while True:
        try:
            last_price = None
            last_ts = None

            if db_pool is not None:
                async with db_pool.acquire() as conn:
                    try:
                        row = await conn.fetchrow(
                            """
                            SELECT close, close_time
                            FROM candles_1m
                            WHERE symbol = 'BTC-USDT-SWAP'
                            ORDER BY close_time DESC
                            LIMIT 1;
                            """
                        )
                        if row:
                            last_price = row["close"]
                            last_ts = row["close_time"]
                    except Exception:
                        pass

            # ---- Pseudo AI logic (very simple, just to have a signal) ----
            signal = "WAIT"
            confidence = 0.0

            if isinstance(last_price, (int, float)):
                # نمط بسيط جداً: إذا السعر زوجي → BUY ، فردي → SELL
                if int(last_price) % 2 == 0:
                    signal = "BUY"
                    confidence = 0.55
                else:
                    signal = "SELL"
                    confidence = 0.55

            logger.info(
                f"AI decision → signal={signal}, confidence={confidence:.2f}, "
                f"price={last_price}, ts={last_ts}"
            )

            # Record into Neon as market_event:
            if writer is not None:
                try:
                    await writer.write_market_event(
                        {
                            "symbol": "BTC-USDT-SWAP",
                            "event_type": "AI_SIGNAL",
                            "event_data": {
                                "signal": signal,
                                "confidence": confidence,
                                "last_price": float(last_price)
                                if isinstance(last_price, (int, float))
                                else None,
                            },
                            "timestamp": last_ts,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to write AI event: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"AI brain loop error: {e}", exc_info=True)

        await asyncio.sleep(interval_sec)


async def dashboard_server_task() -> None:
    """
    Run FastAPI dashboard (if uvicorn is installed).
    The app is in okx_stream_hunter.dashboard.app:app
    """
    logger = logging.getLogger("dashboard")

    try:
        import uvicorn  # type: ignore
    except ImportError:
        logger.warning("uvicorn not installed → dashboard disabled.")
        return

    from okx_stream_hunter.dashboard.app import app as fastapi_app

    host = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    port = int(os.getenv("DASHBOARD_PORT", "8000"))

    config = uvicorn.Config(
        fastapi_app,
        host=host,
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    logger.info(f"Starting dashboard on http://{host}:{port} ...")
    await server.serve()


async def run_stream_engine(
    engine: StreamEngine,
) -> None:
    """
    Wrapper around StreamEngine to ensure clean start/stop.
    """
    logger = logging.getLogger("stream-runner")
    logger.info("Starting StreamEngine ...")
    await engine.start()
    logger.info("StreamEngine started.")

    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        logger.info("StreamEngine task cancelled, shutting down...")
        raise
    finally:
        await engine.stop()
        logger.info("StreamEngine stopped.")


# ============================================================
#  MAIN
# ============================================================

async def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("Loading settings via get_settings() ...")
    settings = get_settings()

    # ----------------------------------------------
    # Database writer (Neon)
    # ----------------------------------------------
    db_writer: Optional[NeonDBWriter] = None
    db_pool: Optional[asyncpg.Pool] = None

    if asyncpg is None:
        logger.warning("asyncpg not available → DB will be disabled.")
    else:
        db_writer = await create_db_writer_if_enabled(settings)
        if db_writer is not None:
            # neon_writer internally يفتح pool خاص به، لكن نقدر نفتح واحد
            # منفصل للقراءة السريعة (analytics / AI).
            try:
                db_pool = await asyncpg.create_pool(
                    dsn=os.getenv("NEON_DATABASE_URL", settings.database.url),
                    min_size=1,
                    max_size=3,
                    command_timeout=60,
                )
                logger.info("Read-only DB pool created for analytics/AI.")
            except Exception as e:
                logger.warning(f"Failed to create read-only DB pool: {e}")
                db_pool = None

    # ----------------------------------------------
    # StreamEngine
    # ----------------------------------------------
    engine_logger = logging.getLogger("stream")
    engine = StreamEngine(
        symbols=settings.okx.symbols,
        channels=settings.okx.channels,
        ws_url=settings.okx.public_ws,
        logger=engine_logger,
        db_writer=db_writer,
    )

    # ----------------------------------------------
    # Background tasks
    # ----------------------------------------------
    tasks: List[asyncio.Task] = []

    tasks.append(asyncio.create_task(run_stream_engine(engine)))
    tasks.append(asyncio.create_task(health_monitor_task(db_pool)))

    # AI Brain
    tasks.append(asyncio.create_task(ai_brain_loop(db_pool, db_writer)))

    # Dashboard (optional, can be disabled via env)
    if os.getenv("ENABLE_DASHBOARD", "1") == "1":
        tasks.append(asyncio.create_task(dashboard_server_task()))
    else:
        logger.info("Dashboard disabled via ENABLE_DASHBOARD=0")

    logger.info("All core tasks started. Beast is running 🐉")

    # ----------------------------------------------
    # Global supervision
    # ----------------------------------------------
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Main cancelled, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
    finally:
        # Cancel all tasks
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        # Close DB writer + pool
        if db_writer is not None:
            try:
                await db_writer.flush_all()
                await db_writer.disconnect()
            except Exception as e:
                logger.warning(f"Error while closing db_writer: {e}")

        if db_pool is not None:
            await db_pool.close()

        logger.info("Shutdown complete. Bye.")


if __name__ == "__main__":
    asyncio.run(main())
