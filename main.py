"""
Entry point for running the real-time OKX Stream Hunter engine.

Usage (on your server, inside the venv):

    python3 main.py

Make sure you have:

    - Installed requirements:  pip install -r requirements.txt
    - Set NEON_DATABASE_URL (or disabled DB via enable_db=False)
    - Optionally set OKX / Claude API keys in environment variables

This script will:
    - Load configuration
    - Start StreamEngine
    - Run until interrupted (Ctrl+C)
"""

import asyncio

from okx_stream_hunter.config.loader import get_config
from okx_stream_hunter.core.engine import StreamEngine
from okx_stream_hunter.utils.logger import get_logger

logger = get_logger(__name__)


async def _run(enable_db: bool = True) -> None:
    cfg = get_config()
    symbol = cfg.get("trading", "symbol", default="BTC-USDT-SWAP")
    timeframes = cfg.get("trading", "timeframes", default=["1m", "5m", "15m"])

    engine = StreamEngine(symbol=symbol, timeframes=timeframes, enable_db=enable_db)

    try:
        await engine.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received – stopping engine...")
        await engine.stop()
    except Exception as e:
        logger.error("Fatal error in main loop: %s", e)
        await engine.stop()


if __name__ == "__main__":
    # If you want to run without DB while testing, set enable_db=False here.
    asyncio.run(_run(enable_db=True))
