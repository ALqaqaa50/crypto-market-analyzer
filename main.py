# main.py
# ============================================================
#  👑 CRYPTO MARKET ANALYZER - BEAST MAIN LAUNCHER
# ============================================================
#  - Loads settings via get_settings()
#  - Connects to Neon (Postgres) using NeonDBWriter
#  - Starts OKX StreamEngine (WS ingestion → DB)
#  - Runs background tasks:
#       * health_monitor_task     → مراقبة صحة الـ DB والجداول
#       * ai_brain_ultra_loop     → AI Brain + Auto-Trade + Insights/Strategy JSON
#       * dashboard_server_task   → FastAPI dashboard على بورت 8000
#  - Single entrypoint:
#       python main.py
# ============================================================

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional, Literal, Dict, Any

try:
    import asyncpg  # type: ignore
except Exception:
    asyncpg = None

from okx_stream_hunter.config.loader import get_settings
from okx_stream_hunter.core import StreamEngine
from okx_stream_hunter.storage.neon_writer import NeonDBWriter
from okx_stream_hunter.core.ai_brain import AIBrain


# ============================================================
#  Helpers
# ============================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def create_db_writer_if_enabled(settings) -> Optional[NeonDBWriter]:
    """
    Create NeonDBWriter only if database is enabled in settings.
    """
    logger = logging.getLogger("db-init")
    db_cfg = settings.database

    if not getattr(db_cfg, "enabled", False):
        logger.info("Database is disabled in settings → running without DB writer.")
        return None

    # env override
    env_url = os.getenv("NEON_DATABASE_URL", "").strip()
    if env_url:
        db_cfg.url = env_url

    if not getattr(db_cfg, "url", "").strip():
        logger.warning(
            "Database is enabled but URL is empty. "
            "Set NEON_DATABASE_URL or database.url in settings."
        )
        return None

    writer = NeonDBWriter()
    await writer.connect()
    logging.getLogger("neon-writer").info("NeonDBWriter connected and ready.")
    return writer


# ============================================================
#  Background: Health Monitor
# ============================================================

async def health_monitor_task(
    db_pool: Optional["asyncpg.Pool"],
    interval_sec: int = 60,
) -> None:
    logger = logging.getLogger("health-monitor")

    if db_pool is None:
        logger.info("DB pool is None → health monitor will only log heartbeat.")
    else:
        logger.info("Health monitor started with DB checks enabled.")

    while True:
        try:
            if db_pool is not None:
                async with db_pool.acquire() as conn:
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
                            counts[table] = None

                    logger.info(f"DB Health: {counts}")
            else:
                logger.info("Health: DB disabled, heartbeat OK.")
        except Exception as e:
            logger.warning(f"Health monitor error: {e}", exc_info=True)

        await asyncio.sleep(interval_sec)


# ============================================================
#  Background: AI Brain + Auto Trading (ULTRA MODE)
# ============================================================

Direction = Literal["long", "short", "flat"]


async def ai_brain_ultra_loop(
    symbol: str,
    db_pool: Optional["asyncpg.Pool"],
    writer: Optional[NeonDBWriter],
    interval_sec: int = 15,
) -> None:
    """
    ULTRA AI LOOP:
      - يقرأ آخر البيانات من Neon (شموع + صفقات + orderbook)
      - يغذي AIBrain بالبيانات (price + trades + orderbook)
      - يبني إشارة تداول متقدمة (اتجاه + ثقة + تفسير)
      - يكتب:
          * insights.json  → للـ Dashboard (/api/ai/insights)
          * strategy.json  → (/api/strategy) مع TP/SL مقترح
          * market_events  → داخل Neon (AI_SIGNAL / AUTO_TRADE_OPEN / CLOSE)
      - يدير مركز واحد بسيط (paper auto-trade) في الذاكرة.
    """
    logger = logging.getLogger("ai-ultra")

    brain = AIBrain(symbol=symbol, logger=logger)

    position: Dict[str, Any] = {
        "direction": "flat",   # "long" / "short" / "flat"
        "size": 0.0,
        "entry_price": None,
        "opened_at": None,
    }

    whale_threshold = float(os.getenv("AI_WHALE_THRESHOLD", "50"))
    base_pos_size = float(os.getenv("AI_BASE_POSITION_SIZE", "0.01"))
    min_trade_conf = float(os.getenv("AI_MIN_TRADE_CONFIDENCE", "0.35"))
    sl_pct = float(os.getenv("AI_SL_PCT", "0.003"))   # 0.3%
    rr_ratio = float(os.getenv("AI_RR_RATIO", "2.0")) # TP:SL = 2:1

    logger.info(
        f"AI ULTRA loop started for {symbol} | base_size={base_pos_size}, "
        f"min_conf={min_trade_conf}, sl_pct={sl_pct}, rr={rr_ratio}"
    )

    while True:
        try:
            last_price: Optional[float] = None
            last_ts: Optional[datetime] = None

            buy_volume = 0.0
            sell_volume = 0.0
            trade_count = 0
            whale_trades = 0
            max_trade_size = 0.0
            max_trade_price: Optional[float] = None

            # -----------------------------
            # Pull data from DB (if enabled)
            # -----------------------------
            if db_pool is not None:
                async with db_pool.acquire() as conn:
                    # 1) شموع 1 دقيقة (آخر 200)
                    candle_rows = []
                    try:
                        candle_rows = await conn.fetch(
                            """
                            SELECT close, close_time
                            FROM candles_1m
                            WHERE symbol = $1
                            ORDER BY close_time DESC
                            LIMIT 200;
                            """,
                            symbol,
                        )
                    except Exception:
                        pass

                    # feed ticker series from oldest → أحدث
                    if candle_rows:
                        for row in reversed(candle_rows):
                            c = float(row["close"])
                            ts = row["close_time"]
                            brain.update_from_ticker(
                                {
                                    "last": c,
                                    "ts": int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000),
                                }
                            )
                        last_price = float(candle_rows[0]["close"])
                        last_ts = candle_rows[0]["close_time"]

                    # 2) آخر الصفقات
                    trade_rows = []
                    try:
                        trade_rows = await conn.fetch(
                            """
                            SELECT side, price, size, timestamp
                            FROM trades
                            WHERE symbol = $1
                            ORDER BY timestamp DESC
                            LIMIT 200;
                            """,
                            symbol,
                        )
                    except Exception:
                        pass

                    trades_for_brain: List[Dict[str, Any]] = []

                    for tr in trade_rows:
                        side = (tr["side"] or "").lower()
                        size = float(tr["size"] or 0)
                        price = float(tr["price"] or 0) if tr["price"] is not None else 0.0
                        ts_dt: datetime = tr["timestamp"]

                        trade_count += 1
                        if side == "buy":
                            buy_volume += size
                        elif side == "sell":
                            sell_volume += size

                        if size >= whale_threshold:
                            whale_trades += 1
                            if size > max_trade_size:
                                max_trade_size = size
                                max_trade_price = price

                        trades_for_brain.append(
                            {
                                "side": side,
                                "size": size,
                                "price": price,
                                "timestamp": int(
                                    ts_dt.replace(tzinfo=timezone.utc).timestamp() * 1000
                                ),
                            }
                        )

                    if trades_for_brain:
                        brain.update_from_trades(trades_for_brain)

                    # 3) آخر orderbook snapshot
                    ob_row = None
                    try:
                        ob_row = await conn.fetchrow(
                            """
                            SELECT best_bid,best_ask,bid_volume,ask_volume,levels_data,timestamp
                            FROM orderbook_snapshots
                            WHERE symbol = $1
                            ORDER BY timestamp DESC
                            LIMIT 1;
                            """,
                            symbol,
                        )
                    except Exception:
                        pass

                    if ob_row:
                        snapshot = {
                            "best_bid": ob_row["best_bid"],
                            "best_ask": ob_row["best_ask"],
                            "bid_volume": ob_row["bid_volume"],
                            "ask_volume": ob_row["ask_volume"],
                            "levels_data": ob_row["levels_data"],
                        }
                        brain.update_from_orderbook(snapshot)

            # -----------------------------
            # Build AI signal
            # -----------------------------
            sig = brain.build_signal()
            direction: Direction = sig["direction"]
            confidence: float = float(sig["confidence"])
            reason: str = sig.get("reason", "")
            regime: str = sig.get("regime", "unknown")
            price_for_decision: Optional[float] = (
                float(sig["price"]) if sig.get("price") is not None else last_price
            )

            logging.getLogger("ai-ultra").info(
                f"AI SIGNAL → dir={direction}, conf={confidence:.3f}, "
                f"regime={regime}, reason={reason}, price={price_for_decision}"
            )

            # -----------------------------
            # Auto-Trade Logic (Paper)
            # -----------------------------
            auto_event: Optional[Dict[str, Any]] = None
            trade_signal_str = "WAIT"

            if direction == "long":
                trade_signal_str = "BUY"
            elif direction == "short":
                trade_signal_str = "SELL"
            else:
                trade_signal_str = "WAIT"

            if (
                direction in ("long", "short")
                and confidence >= min_trade_conf
                and price_for_decision is not None
            ):
                # open / flip / manage
                if position["direction"] == "flat":
                    # open new
                    position["direction"] = direction
                    position["size"] = base_pos_size
                    position["entry_price"] = price_for_decision
                    position["opened_at"] = datetime.now(timezone.utc)

                    auto_event = {
                        "symbol": symbol,
                        "event_type": "AUTO_TRADE_OPEN",
                        "event_data": {
                            "direction": direction,
                            "size": base_pos_size,
                            "entry_price": price_for_decision,
                            "confidence": confidence,
                            "reason": reason,
                        },
                        "timestamp": position["opened_at"],
                    }

                elif position["direction"] != direction:
                    # flip: close then open opposite (as two events)
                    close_event = {
                        "symbol": symbol,
                        "event_type": "AUTO_TRADE_CLOSE",
                        "event_data": {
                            "prev_direction": position["direction"],
                            "prev_size": position["size"],
                            "entry_price": position["entry_price"],
                            "close_price": price_for_decision,
                            "opened_at": position["opened_at"],
                            "closed_at": datetime.now(timezone.utc),
                        },
                        "timestamp": datetime.now(timezone.utc),
                    }

                    # open new
                    position["direction"] = direction
                    position["size"] = base_pos_size
                    position["entry_price"] = price_for_decision
                    position["opened_at"] = datetime.now(timezone.utc)

                    open_event = {
                        "symbol": symbol,
                        "event_type": "AUTO_TRADE_OPEN",
                        "event_data": {
                            "direction": direction,
                            "size": base_pos_size,
                            "entry_price": price_for_decision,
                            "confidence": confidence,
                            "reason": reason,
                        },
                        "timestamp": position["opened_at"],
                    }

                    if writer is not None:
                        try:
                            await writer.write_market_event(close_event)
                            await writer.write_market_event(open_event)
                        except Exception as e:
                            logger.warning(f"Failed to write flip events: {e}", exc_info=True)

                    auto_event = None  # already written
                # else: same direction already open → لا نفتح من جديد

            else:
                # if direction flat أو ثقة ضعيفة → ممكن نغلق مركز مفتوح
                if position["direction"] != "flat" and price_for_decision is not None:
                    close_event = {
                        "symbol": symbol,
                        "event_type": "AUTO_TRADE_CLOSE",
                        "event_data": {
                            "prev_direction": position["direction"],
                            "prev_size": position["size"],
                            "entry_price": position["entry_price"],
                            "close_price": price_for_decision,
                            "opened_at": position["opened_at"],
                            "closed_at": datetime.now(timezone.utc),
                            "reason": "signal_flat_or_low_conf",
                        },
                        "timestamp": datetime.now(timezone.utc),
                    }

                    if writer is not None:
                        try:
                            await writer.write_market_event(close_event)
                        except Exception as e:
                            logger.warning(f"Failed to write close event: {e}", exc_info=True)

                    position = {
                        "direction": "flat",
                        "size": 0.0,
                        "entry_price": None,
                        "opened_at": None,
                    }

            if auto_event is not None and writer is not None:
                try:
                    await writer.write_market_event(auto_event)
                except Exception as e:
                    logger.warning(f"Failed to write auto_event: {e}", exc_info=True)

            # -----------------------------
            # Write insights.json + strategy.json
            # -----------------------------
            now_iso = datetime.now(timezone.utc).isoformat()

            insights_payload = {
                "signal": trade_signal_str,
                "direction": direction,
                "confidence": round(confidence * 100, 1),  # 0-100
                "reason": reason,
                "regime": regime,
                "price": price_for_decision,
                "buy_volume": round(buy_volume, 4),
                "sell_volume": round(sell_volume, 4),
                "trade_count": trade_count,
                "whale_trades": whale_trades,
                "max_trade_size": max_trade_size,
                "max_trade_price": max_trade_price,
                "position": {
                    "direction": position["direction"],
                    "size": position["size"],
                    "entry_price": position["entry_price"],
                },
                "scores": sig.get("scores", {}),
                "generated_at": now_iso,
            }

            strategy_entry = price_for_decision or last_price
            tp = None
            sl = None
            if strategy_entry is not None and direction in ("long", "short"):
                if direction == "long":
                    sl = strategy_entry * (1.0 - sl_pct)
                    tp = strategy_entry * (1.0 + sl_pct * rr_ratio)
                else:
                    sl = strategy_entry * (1.0 + sl_pct)
                    tp = strategy_entry * (1.0 - sl_pct * rr_ratio)

            strategy_payload = {
                "signal": trade_signal_str,
                "direction": direction,
                "entry_price": strategy_entry,
                "tp": tp,
                "sl": sl,
                "confidence": round(confidence * 100, 1),
                "generated_at": now_iso,
            }

            # write JSON files (for FastAPI dashboard)
            try:
                with open("insights.json", "w", encoding="utf-8") as f:
                    json.dump(insights_payload, f, ensure_ascii=False, indent=2)
                with open("strategy.json", "w", encoding="utf-8") as f:
                    json.dump(strategy_payload, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write insights/strategy JSON: {e}", exc_info=True)

            # also log AI_SIGNAL into DB
            if writer is not None:
                try:
                    await writer.write_market_event(
                        {
                            "symbol": symbol,
                            "event_type": "AI_SIGNAL",
                            "event_data": {
                                "direction": direction,
                                "signal": trade_signal_str,
                                "confidence": confidence,
                                "regime": regime,
                                "reason": reason,
                                "price": price_for_decision,
                            },
                            "timestamp": datetime.now(timezone.utc),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to write AI_SIGNAL event: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"AI ULTRA loop error: {e}", exc_info=True)

        await asyncio.sleep(interval_sec)


# ============================================================
#  Dashboard Task
# ============================================================

async def dashboard_server_task() -> None:
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


# ============================================================
#  Stream Engine Runner
# ============================================================

async def run_stream_engine(engine: StreamEngine) -> None:
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

    target_symbol = (
        settings.okx.symbols[0]
        if getattr(settings.okx, "symbols", None)
        else "BTC-USDT-SWAP"
    )
    logger.info(f"Target symbol for AI/Stream: {target_symbol}")

    # ----------------- DB Writer & Pool -----------------
    db_writer: Optional[NeonDBWriter] = None
    db_pool: Optional["asyncpg.Pool"] = None

    if asyncpg is None:
        logger.warning("asyncpg not available → DB features disabled.")
    else:
        db_writer = await create_db_writer_if_enabled(settings)
        if db_writer is not None:
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

    # ----------------- StreamEngine -----------------
    engine_logger = logging.getLogger("stream")
    engine = StreamEngine(
        symbols=settings.okx.symbols,
        channels=settings.okx.channels,
        ws_url=settings.okx.public_ws,
        logger=engine_logger,
        db_writer=db_writer,
    )

    tasks: List[asyncio.Task] = []
    tasks.append(asyncio.create_task(run_stream_engine(engine)))
    tasks.append(asyncio.create_task(health_monitor_task(db_pool)))

    # AI ULTRA MODE
    tasks.append(
        asyncio.create_task(
            ai_brain_ultra_loop(
                symbol=target_symbol,
                db_pool=db_pool,
                writer=db_writer,
            )
        )
    )

    # Dashboard
    if os.getenv("ENABLE_DASHBOARD", "1") == "1":
        tasks.append(asyncio.create_task(dashboard_server_task()))
    else:
        logger.info("Dashboard disabled via ENABLE_DASHBOARD=0")

    logger.info("All core tasks started. Beast is running 🐉")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Main cancelled, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

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
