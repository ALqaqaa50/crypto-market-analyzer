# main.py

import asyncio
import logging
import os
from typing import Optional

from okx_stream_hunter.config.loader import load_settings
from okx_stream_hunter.core import StreamEngine
from okx_stream_hunter.storage.neon_writer import NeonDBWriter


async def create_db_writer_if_enabled(settings) -> Optional[NeonDBWriter]:
    db_cfg = settings.database
    if not db_cfg.enabled:
        return None

    if not db_cfg.url:
        logging.getLogger("main").warning(
            "Database is enabled but URL is empty. Skipping DB writer initialization."
        )
        return None

    logger = logging.getLogger("db")
    writer = NeonDBWriter(
        dsn=db_cfg.url,
        min_size=db_cfg.pool_min_size,
        max_size=db_cfg.pool_max_size,
        logger=logger,
    )
    await writer.initialize()
    return writer


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger = logging.getLogger("main")
    logger.info("Loading settings...")
    settings = load_settings("config/settings.yaml")

    # Ensure env expansion (NEON_DATABASE_URL etc.) via loader
    if not settings.database.url:
        # fallback to raw env if needed
        env_url = os.getenv("NEON_DATABASE_URL", "")
        if env_url:
            settings.database.url = env_url

    db_writer = await create_db_writer_if_enabled(settings)

    engine_logger = logging.getLogger("stream")
    engine = StreamEngine(
        symbols=settings.okx.symbols,
        channels=settings.okx.channels,
        ws_url=settings.okx.public_ws,
        logger=engine_logger,
        db_writer=db_writer,
    )

    await engine.start()
    logger.info("StreamEngine started.")

    try:
        while True:
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
    finally:
        await engine.stop()
        if db_writer:
            await db_writer.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
