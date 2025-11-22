# okx_stream_hunter/core/processor.py

import asyncio
from typing import Any, Dict, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProcessedMessage:
    """
    Normalized market message.

    type: 'trade' | 'candle' | 'orderbook' | 'system'
    payload: original data dict
    """

    __slots__ = ("type", "symbol", "data")

    def __init__(self, msg_type: str, symbol: str, data: Dict[str, Any]) -> None:
        self.type = msg_type
        self.symbol = symbol
        self.data = data

    def __repr__(self) -> str:  # for debugging
        return f"<ProcessedMessage type={self.type} symbol={self.symbol}>"


class MarketProcessor:
    """
    Consume raw WS messages and produce normalized ProcessedMessage objects.

    Design:
    - Reads from `raw_queue`
    - Decodes OKX format (arg.channel, arg.instId, data)
    - Routes into:
        - trades_queue
        - candles_queue
        - orderbook_queue
    """

    def __init__(
        self,
        raw_queue: asyncio.Queue,
        trades_queue: Optional[asyncio.Queue] = None,
        candles_queue: Optional[asyncio.Queue] = None,
        orderbook_queue: Optional[asyncio.Queue] = None,
    ) -> None:
        self.raw_queue = raw_queue
        self.trades_queue = trades_queue or asyncio.Queue(maxsize=20_000)
        self.candles_queue = candles_queue or asyncio.Queue(maxsize=20_000)
        self.orderbook_queue = orderbook_queue or asyncio.Queue(maxsize=5_000)

        self._task: Optional[asyncio.Task] = None
        self._running = False

    # -------------------- public API --------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="market-processor")
        logger.info("MarketProcessor started")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("MarketProcessor stopped")

    # -------------------- internals --------------------

    async def _run(self) -> None:
        while self._running:
            try:
                msg = await self.raw_queue.get()
            except asyncio.CancelledError:
                break

            try:
                processed = self._normalize(msg)
                if processed is None:
                    continue
                await self._dispatch(processed)
            except Exception as e:
                logger.error("Processor failed on message %s: %s", msg, e, exc_info=True)

    def _normalize(self, msg: Dict[str, Any]) -> Optional[ProcessedMessage]:
        """
        Convert OKX WS payload into ProcessedMessage.

        Expected structure:
        {
          "arg": {"channel": "...", "instId": "..."},
          "data": [...]
        }
        """
        if not isinstance(msg, dict):
            return None

        arg = msg.get("arg") or {}
        channel = arg.get("channel")
        symbol = arg.get("instId", "")

        if not channel:
            # system notification / login / etc.
            return ProcessedMessage("system", symbol or "system", msg)

        data = msg.get("data") or []

        # trades
        if channel.endswith("trades") or channel == "trades":
            return ProcessedMessage("trade", symbol, {"raw": data, "channel": channel})

        # candles: e.g. "candle1m", "candle15m"
        if channel.startswith("candle"):
            timeframe = channel.replace("candle", "")
            return ProcessedMessage(
                "candle", symbol, {"raw": data, "timeframe": timeframe}
            )

        # orderbook, e.g. books5, books, books-l2-tbt
        if channel.startswith("books"):
            return ProcessedMessage(
                "orderbook", symbol, {"raw": data, "channel": channel}
            )

        # otherwise keep as system
        return ProcessedMessage("system", symbol or "system", msg)

    async def _dispatch(self, msg: ProcessedMessage) -> None:
        """Send the processed message to the appropriate queue."""
        if msg.type == "trade":
            await self._safe_put(self.trades_queue, msg)
        elif msg.type == "candle":
            await self._safe_put(self.candles_queue, msg)
        elif msg.type == "orderbook":
            await self._safe_put(self.orderbook_queue, msg)
        else:
            # system messages are not queued separately for now
            logger.debug("System message: %s", msg)

    @staticmethod
    async def _safe_put(q: asyncio.Queue, item: Any) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # fall back to awaited put (backpressure)
            await q.put(item)
