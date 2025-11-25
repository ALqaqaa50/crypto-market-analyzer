# okx_stream_hunter/core/stream_manager.py

import asyncio
import logging
from typing import Dict, List, Optional

from okx_stream_hunter.core.processor import MarketProcessor
from okx_stream_hunter.core.supervisor import StreamSupervisor
from okx_stream_hunter.core.ws_client import OKXWebSocketClient
from okx_stream_hunter.storage.neon_writer import NeonDBWriter


class StreamEngine:
    """
    Coordinates:
    - WebSocket client
    - MarketProcessor
    - Supervisor for health
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        ws_url: str = "wss://ws.okx.com:8443/ws/v5/public",
        db_writer: Optional[NeonDBWriter] = None,
        ai_brain=None,
    ) -> None:

        self.logger = logger or logging.getLogger(__name__)

        self.symbols = symbols or ["BTC-USDT-SWAP"]
        self.channels = channels or ["trades", "books5", "tickers"]
        self.ws_url = ws_url

        # Processor
        self.processor = MarketProcessor(
            logger=self.logger.getChild("processor"),
            db_writer=db_writer,
            db_enable_trades=True,
            db_enable_orderbook=True,
            ai_brain=ai_brain,
        )

        # Build subscription arguments
        subscriptions = self._build_subscriptions()

        # WebSocket client
        self.ws_client = OKXWebSocketClient(
            url=self.ws_url,
            subscriptions=subscriptions,
            on_message=self.processor.handle_message,
            logger=self.logger.getChild("ws"),
        )

        # Supervisor يقلّم آخر رسالة وصلت من WebSocket مباشرة
        self.supervisor = StreamSupervisor(
            get_last_update_ts=lambda: self.ws_client.last_message_ts,
            logger=self.logger.getChild("supervisor"),
            on_warning=self._handle_warning,
            check_interval=5,
            stale_after=25,   # تم تقليلها لأن ws_client يرسل ping/pong
        )

        self._tasks: List[asyncio.Task] = []
        self._running = False

    # ------------------------------------------------------------------
    def _build_subscriptions(self) -> List[Dict[str, str]]:
        args: List[Dict[str, str]] = []
        for inst in self.symbols:
            for ch in self.channels:
                args.append({"channel": ch, "instId": inst})
        return args

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self.logger.info(
            f"Starting StreamEngine with symbols={self.symbols}, channels={self.channels}"
        )

        # WebSocket task
        ws_task = asyncio.create_task(self.ws_client.start(), name="okx-ws-client")
        self._tasks.append(ws_task)

        # Supervisor task
        sup_task = asyncio.create_task(self.supervisor.start(), name="stream-supervisor")
        self._tasks.append(sup_task)

        self.logger.info("StreamEngine started.")

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        self.logger.info("Stopping StreamEngine...")

        await self.ws_client.stop()
        await self.supervisor.stop()

        for t in self._tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        self.logger.info("StreamEngine stopped.")

    # ------------------------------------------------------------------
    async def _handle_warning(self, message: str) -> None:
        self.logger.warning(f"[SUPERVISOR WARNING] {message}")
