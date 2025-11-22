# okx_stream_hunter/core/stream_manager.py

import asyncio
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from .ws_client import WSClient
from .processor import MarketProcessor

logger = get_logger(__name__)


class StreamEngine:
    """
    High level streaming engine.

    - Owns a WSClient for OKX
    - Owns a MarketProcessor for message normalization
    - Exposes public queues: trades, candles, orderbook
    """

    def __init__(
        self,
        *,
        url: str,
        subscriptions: List[Dict[str, Any]],
        name: str = "okx-stream-engine",
    ) -> None:
        # raw messages from WS
        self.raw_queue: asyncio.Queue = asyncio.Queue(maxsize=20_000)

        # core components
        self.ws_client = WSClient(
            url=url,
            subscriptions=subscriptions,
            name=name + "-ws",
            max_queue_size=20_000,
        )
        # wire ws_client queue to our raw_queue
        self.ws_client._queue = self.raw_queue  # internal wiring, same queue object

        self.processor = MarketProcessor(self.raw_queue)

        self.name = name
        self._running = False

    # Convenience properties to access processed queues
    @property
    def trades_queue(self) -> asyncio.Queue:
        return self.processor.trades_queue

    @property
    def candles_queue(self) -> asyncio.Queue:
        return self.processor.candles_queue

    @property
    def orderbook_queue(self) -> asyncio.Queue:
        return self.processor.orderbook_queue

    # -------------------- lifecycle --------------------

    async def start(self) -> None:
        """Start WS client + processor."""
        if self._running:
            return
        self._running = True
        await self.ws_client.start()
        await self.processor.start()
        logger.info("StreamEngine '%s' started", self.name)

    async def stop(self) -> None:
        """Stop all components."""
        self._running = False
        await self.processor.stop()
        await self.ws_client.stop()
        logger.info("StreamEngine '%s' stopped", self.name)

    # -------------------- helper methods --------------------

    async def drain_once(self) -> Dict[str, int]:
        """
        Utility to drain queues once for debugging / testing.

        Returns simple stats dict with number of items consumed from each queue.
        """
        stats = {"trades": 0, "candles": 0, "orderbook": 0}

        while not self.trades_queue.empty():
            _ = await self.trades_queue.get()
            stats["trades"] += 1
        while not self.candles_queue.empty():
            _ = await self.candles_queue.get()
            stats["candles"] += 1
        while not self.orderbook_queue.empty():
            _ = await self.orderbook_queue.get()
            stats["orderbook"] += 1

        return stats
