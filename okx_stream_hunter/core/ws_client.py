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
    Ultra-Stable WebSocket client for OKX streams.

    - Auto reconnect
    - Heartbeat (ping/pong) handling
    - Tracks last message timestamp
    - Clean subscribe/unsubscribe API
    - Fully compatible with supervisor
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

        # Supervisor يستخدمها لتحديد stale stream
        self._last_message_ts: float = time.time()

    # ------------------------------------------------------------------
    @property
    def last_message_ts(self) -> float:
        return self._last_message_ts

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._stopping = False
        attempt = 0

        while not self._stopping:
            try:
                await self._connect_and_run()
                attempt = 0
            except asyncio.CancelledError:
                self.logger.info("WebSocket client cancelled.")
                break
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}", exc_info=True)
                attempt += 1

                if 0 < self._max_retries < attempt:
                    self.logger.error("Max retries reached. Stopping client.")
                    break

                backoff = min(
                    self._base_backoff * (2 ** (attempt - 1)),
                    self._max_backoff,
                )
                self.logger.info(f"Reconnecting in {backoff:.1f}s...")
                await asyncio.sleep(backoff)

        self._running = False
        await self._close()

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        self._stopping = True
        await self._close()

    # ------------------------------------------------------------------
    async def subscribe(self, args: List[Dict[str, str]]) -> None:
        self.subscriptions.extend(args)
        if not self.connected:
            return
        await self._send({"op": "subscribe", "args": args})

    async def unsubscribe(self, args: List[Dict[str, str]]) -> None:
        if not self.connected:
            return
        await self._send({"op": "unsubscribe", "args": args})

    # ------------------------------------------------------------------
    async def _connect_and_run(self) -> None:

        self.logger.info(f"Connecting to OKX WebSocket: {self.url}")

        async with websockets.connect(
            self.url,
            ping_interval=15,   # إرسال ping كل 15 ثانية
            ping_timeout=10,    # timeout للـ pong
            max_size=10_000_000
        ) as ws:

            self._ws = ws
            self.logger.info("WebSocket connected.")

            await self._subscribe_all()

            async for raw in ws:
                self._last_message_ts = time.time()

                # Ping/Pong from OKX
                if raw == "pong":
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"Decode failed: {raw!r}")
                    continue

                # Events: subscribe + error
                if "event" in data:
                    self._handle_event(data)
                    continue

                if self.on_message:
                    try:
                        await self.on_message(data)
                    except Exception as e:
                        self.logger.error(f"Handler error: {e}", exc_info=True)

    # ------------------------------------------------------------------
    async def _subscribe_all(self) -> None:
        if not self.subscriptions:
            return
        self.logger.info(f"Subscribing to {len(self.subscriptions)} streams.")
        await self._send({"op": "subscribe", "args": self.subscriptions})

    # ------------------------------------------------------------------
    async def _send(self, msg: Dict[str, Any]) -> None:
        if not self._ws or self._ws.closed:
            self.logger.warning(f"Cannot send, WS disconnected: {msg}")
            return
        await self._ws.send(json.dumps(msg))

    # ------------------------------------------------------------------
    async def _close(self) -> None:
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
                self.logger.info("WebSocket closed.")
            except Exception as e:
                self.logger.warning(f"Close error: {e}")
        self._ws = None

    # ------------------------------------------------------------------
    def _handle_event(self, data: Dict[str, Any]) -> None:

        event = data.get("event")
        arg = data.get("arg") or data.get("args")

        if event == "subscribe":
            self.logger.info(f"Subscribed: {arg}")
        elif event == "error":
            code = data.get("code")
            msg = data.get("msg")
            self.logger.error(f"OKX Error: code={code}, msg={msg}")
        else:
            self.logger.debug(f"WS Event: {data}")
