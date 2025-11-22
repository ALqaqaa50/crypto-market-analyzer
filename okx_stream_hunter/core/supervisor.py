# okx_stream_hunter/core/supervisor.py

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable

from ..utils.logger import get_logger
from .stream_manager import StreamEngine

logger = get_logger(__name__)


class StreamSupervisor:
    """
    Supervisor around StreamEngine.

    - Ensures the engine is running
    - Restarts it on fatal failure with backoff
    - Exposes hooks for health checks and notifications
    """

    def __init__(
        self,
        engine_factory: Callable[[], StreamEngine],
        *,
        name: str = "okx-stream-supervisor",
        restart_base_delay: float = 5.0,
        restart_max_delay: float = 120.0,
    ) -> None:
        self.engine_factory = engine_factory
        self.name = name
        self.restart_base_delay = restart_base_delay
        self.restart_max_delay = restart_max_delay

        self._engine: Optional[StreamEngine] = None
        self._task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def engine(self) -> Optional[StreamEngine]:
        return self._engine

    # -------------------- lifecycle --------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name=self.name)
        logger.info("StreamSupervisor '%s' started", self.name)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        if self._engine:
            await self._engine.stop()
        logger.info("StreamSupervisor '%s' stopped", self.name)

    # -------------------- internals --------------------

    async def _run(self) -> None:
        delay = self.restart_base_delay

        while self._running:
            try:
                self._engine = self.engine_factory()
                logger.info("StreamSupervisor '%s' starting engine", self.name)
                await self._engine.start()

                # Wait forever until cancelled or error
                while self._running:
                    await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                logger.info("StreamSupervisor '%s' cancelled", self.name)
                break
            except Exception as e:
                logger.error(
                    "StreamSupervisor '%s' caught engine error: %s",
                    self.name,
                    e,
                    exc_info=True,
                )

            # engine crashed or supervisor is stopping
            if not self._running:
                break

            logger.warning(
                "StreamSupervisor '%s' restarting engine in %.1f seconds",
                self.name,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.restart_max_delay)
