# okx_stream_hunter/core/ws_client.py

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import websockets
from websockets import WebSocketClientProtocol


MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class OKXWebSocketClient:
    """
    Robust WebSocket client for OKX public streams.

    - Auto reconnect with exponential backoff
    - Heartbeat (ping/pong) handling
    - Graceful shutdown
    - Simple subscribe/unsubscribe API
    """

    def __init__(
        self,
        url: str,
        subscriptions: Optional[List[Dict[str, str]]] = None,
        on_message: Optional[MessageHandler] = None,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 0,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.url = url
        self.subscriptions = subscriptions or []
        self.on_message = on_message
        self.logger = logger or logging.getLogger(__name__)

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._stopping = False

        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff

        self._last_message_ts: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def last_message_ts(self) -> float:
        return self._last_message_ts

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """
        Start the WebSocket client and keep it running
        with automatic reconnects.
        """
        if self._running:
            return

        self._running = True
        self._stopping = False
        attempt = 0

        while not self._stopping:
            try:
                await self._connect_and_run()
                attempt = 0  # reset on successful session
            except asyncio.CancelledError:
                self.logger.info("WebSocket client cancelled, stopping.")
                break
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}", exc_info=True)
                attempt += 1

                if 0 < self._max_retries < attempt:
                    self.logger.error(
                        "Max retries exceeded, stopping WebSocket client."
                    )
                    break

                backoff = min(self._base_backoff * (2 ** (attempt - 1)), self._max_backoff)
                self.logger.info(f"Reconnecting in {backoff:.1f} seconds...")
                await asyncio.sleep(backoff)

        self._running = False
        await self._close()

    async def stop(self) -> None:
        """Request graceful shutdown."""
        self._stopping = True
        await self._close()

    async def subscribe(self, args: List[Dict[str, str]]) -> None:
        """Send subscribe request (and remember it)."""
        self.subscriptions.extend(args)
        if not self.connected:
            return

        msg = {"op": "subscribe", "args": args}
        await self._send(msg)

    async def unsubscribe(self, args: List[Dict[str, str]]) -> None:
        """Send unsubscribe request."""
        if not self.connected:
            return
        msg = {"op": "unsubscribe", "args": args}
        await self._send(msg)

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    async def _connect_and_run(self) -> None:
        self.logger.info(f"Connecting to OKX WebSocket: {self.url}")
        async with websockets.connect(self.url, ping_interval=20, ping_timeout=10) as ws:
            self._ws = ws
            self.logger.info("WebSocket connected.")

            if self.subscriptions:
                await self._subscribe_all()

            async for raw in ws:
                self._last_message_ts = time.time()

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to decode message: {raw!r}")
                    continue

                # OKX sends "event" messages for subscription, login, etc.
                if "event" in data:
                    self._handle_event(data)
                    continue

                if self.on_message:
                    try:
                        await self.on_message(data)
                    except Exception as e:
                        self.logger.error(
                            f"Error in on_message handler: {e}", exc_info=True
                        )

    async def _subscribe_all(self) -> None:
        if not self.subscriptions:
            return
        self.logger.info(f"Subscribing to {len(self.subscriptions)} streams.")
        msg = {"op": "subscribe", "args": self.subscriptions}
        await self._send(msg)

    async def _send(self, msg: Dict[str, Any]) -> None:
        if not self._ws or self._ws.closed:
            self.logger.warning(f"Cannot send, websocket not connected: {msg}")
            return
        raw = json.dumps(msg)
        await self._ws.send(raw)

    async def _close(self) -> None:
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
                self.logger.info("WebSocket connection closed.")
            except Exception as e:
                self.logger.warning(f"Error while closing WebSocket: {e}")
        self._ws = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _handle_event(self, data: Dict[str, Any]) -> None:
        """
        Handle OKX event messages (subscribe, error, etc.).
        Example:
        {"event":"subscribe","arg":{"channel":"trades","instId":"BTC-USDT-SWAP"}}
        """
        event = data.get("event")
        arg = data.get("arg") or data.get("args")
        if event == "subscribe":
            self.logger.info(f"Subscribed: {arg}")
        elif event == "error":
            code = data.get("code")
            msg = data.get("msg")
            self.logger.error(f"OKX WS error event: code={code}, msg={msg}")
        else:
            self.logger.debug(f"WS event: {data}")
