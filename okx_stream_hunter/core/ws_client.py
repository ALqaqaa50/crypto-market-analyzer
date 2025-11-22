# okx_stream_hunter/core/ws_client.py

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

import websockets
from websockets import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from ..utils.logger import get_logger

logger = get_logger(__name__)


class WSClient:
    """
    Robust WebSocket client for OKX streaming.

    Responsibilities:
    - Maintain a single WS connection
    - Auto-reconnect with backoff
    - Handle ping / pong heartbeats
    - Send subscriptions on (re)connect
    - Push all messages into an asyncio.Queue
    """

    def __init__(
        self,
        url: str,
        subscriptions: List[Dict[str, Any]],
        *,
        name: str = "okx-ws",
        heartbeat_interval: int = 20,
        reconnect_base_delay: float = 5.0,
        reconnect_max_delay: float = 60.0,
        max_queue_size: int = 10_000,
    ) -> None:
        self.url = url
        self.subscriptions = subscriptions
        self.name = name
        self.heartbeat_interval = heartbeat_interval
        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay

        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=max_queue_size
        )
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    # -------------------- public API --------------------

    @property
    def queue(self) -> asyncio.Queue:
        """Queue that will receive raw WS messages as dicts."""
        return self._queue

    async def start(self) -> None:
        """Start the client loop in background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name=f"{self.name}-loop")
        logger.info("WSClient '%s' started", self.name)

    async def stop(self) -> None:
        """Stop the client and close connection."""
        self._running = False
        if self._task:
            self._task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        logger.info("WSClient '%s' stopped", self.name)

    # -------------------- internals --------------------

    async def _run(self) -> None:
        """Main reconnect loop."""
        backoff = self.reconnect_base_delay

        while self._running:
            try:
                await self._connect_and_stream()
                # if we exit gracefully, reset backoff
                backoff = self.reconnect_base_delay
            except asyncio.CancelledError:
                logger.info("WSClient '%s' loop cancelled", self.name)
                break
            except Exception as e:
                logger.error("WSClient '%s' error: %s", self.name, e, exc_info=True)

            if not self._running:
                break

            logger.warning(
                "WSClient '%s' reconnecting in %.1f seconds", self.name, backoff
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.reconnect_max_delay)

    async def _connect_and_stream(self) -> None:
        """Connect to WS, subscribe, then read until connection closes."""
        logger.info("WSClient '%s' connecting to %s", self.name, self.url)
        async with websockets.connect(self.url, ping_interval=None) as ws:
            self._ws = ws
            logger.info("WSClient '%s' connected", self.name)

            # send subscriptions
            await self._send_subscriptions()

            # start heartbeat + reader
            reader = asyncio.create_task(self._reader_loop(), name=f"{self.name}-recv")
            heartbeat = asyncio.create_task(
                self._heartbeat_loop(), name=f"{self.name}-heartbeat"
            )

            done, pending = await asyncio.wait(
                {reader, heartbeat},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()

            # check for exceptions
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc

    async def _send_subscriptions(self) -> None:
        """Send all subscription payloads."""
        if not self._ws:
            return

        for sub in self.subscriptions:
            msg = json.dumps(sub)
            await self._ws.send(msg)
            logger.info("WSClient '%s' subscribed: %s", self.name, msg)

    async def _heartbeat_loop(self) -> None:
        """Send periodic ping frames to keep the connection alive."""
        assert self._ws is not None
        while self._running and self._ws.open:
            try:
                ping_payload = json.dumps({"op": "ping", "ts": int(time.time() * 1000)})
                await self._ws.send(ping_payload)
                await asyncio.sleep(self.heartbeat_interval)
            except (ConnectionClosedError, ConnectionClosedOK):
                logger.warning("WSClient '%s' heartbeat connection closed", self.name)
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("WSClient '%s' heartbeat error: %s", self.name, e)
                break

    async def _reader_loop(self) -> None:
        """Read messages from WS and push them into queue."""
        assert self._ws is not None

        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except Exception:
                logger.warning("WSClient '%s' invalid JSON: %s", self.name, raw)
                continue

            # skip pure pong
            if isinstance(msg, dict) and msg.get("event") == "pong":
                continue

            try:
                self._queue.put_nowait(msg)
            except asyncio.QueueFull:
                logger.error(
                    "WSClient '%s' queue is full; dropping message", self.name
                )
