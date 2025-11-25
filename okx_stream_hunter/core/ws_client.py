# okx_stream_hunter/core/ws_client.py

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional
from collections import deque
from enum import Enum
from datetime import datetime

import websockets
from websockets import WebSocketClientProtocol

MessageHandler = Callable[[Dict[str, Any]], Awaitable[None]]


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    STOPPED = "stopped"


class OKXWebSocketClient:
    """
    PHASE 3: Unbreakable WebSocket Client for OKX
    
    Features:
    - Exponential backoff reconnection (1s → 32s max)
    - Connection state machine
    - Ping/Pong heartbeat monitoring
    - Message queueing during reconnection
    - Auto-resubscribe after reconnect
    - Dead connection detection
    - Connection resilience testing
    """

    def __init__(
        self,
        url: str,
        subscriptions: Optional[List[Dict[str, str]]] = None,
        on_message: Optional[MessageHandler] = None,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 0,
        base_backoff: float = 1.0,
        max_backoff: float = 32.0,
        ping_interval: float = 15.0,
        ping_timeout: float = 10.0,
        dead_connection_threshold: float = 60.0,
    ) -> None:

        self.url = url
        self.subscriptions = subscriptions or []
        self.on_message = on_message
        self.logger = logger or logging.getLogger(__name__)

        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._stopping = False
        
        self._state = ConnectionState.DISCONNECTED

        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._dead_connection_threshold = dead_connection_threshold

        self._last_message_ts: float = time.time()
        self._last_ping_ts: float = time.time()
        self._last_pong_ts: float = time.time()
        
        self._message_queue = deque(maxlen=1000)
        
        self._connection_attempts = 0
        self._successful_connections = 0
        self._total_reconnects = 0
        self._total_messages = 0
        
        self._ping_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    @property
    def last_message_ts(self) -> float:
        return self._last_message_ts

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed and self._state == ConnectionState.CONNECTED
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def stats(self) -> Dict:
        """Get connection statistics"""
        return {
            'state': self._state.value,
            'connection_attempts': self._connection_attempts,
            'successful_connections': self._successful_connections,
            'total_reconnects': self._total_reconnects,
            'total_messages': self._total_messages,
            'last_message_age': time.time() - self._last_message_ts,
            'last_ping_age': time.time() - self._last_ping_ts,
            'last_pong_age': time.time() - self._last_pong_ts,
            'queue_size': len(self._message_queue),
            'subscriptions': len(self.subscriptions)
        }
    
    def is_alive(self) -> bool:
        """Check if connection is alive"""
        if not self.connected:
            return False
        
        message_age = time.time() - self._last_message_ts
        return message_age < self._dead_connection_threshold

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start WebSocket client with unbreakable reconnection"""
        if self._running:
            return

        self._running = True
        self._stopping = False
        self._state = ConnectionState.CONNECTING
        attempt = 0

        self.logger.info("🚀 Starting PHASE 3 Unbreakable WebSocket Client")

        while not self._stopping:
            try:
                self._connection_attempts += 1
                self._state = ConnectionState.CONNECTING if attempt == 0 else ConnectionState.RECONNECTING
                
                if attempt > 0:
                    self.logger.info(f"🔄 Reconnection attempt #{attempt}")
                
                await self._connect_and_run()
                
                attempt = 0
                self._successful_connections += 1
                
            except asyncio.CancelledError:
                self.logger.info("WebSocket client cancelled.")
                break
            except Exception as e:
                self.logger.error(f"❌ WebSocket error: {e}", exc_info=True)
                attempt += 1
                self._total_reconnects += 1

                if 0 < self._max_retries < attempt:
                    self.logger.error(f"🛑 Max retries ({self._max_retries}) reached. Stopping.")
                    break

                backoff = min(
                    self._base_backoff * (2 ** (attempt - 1)),
                    self._max_backoff,
                )
                self.logger.warning(f"⏳ Reconnecting in {backoff:.1f}s... (attempt {attempt})")
                
                self._state = ConnectionState.RECONNECTING
                await asyncio.sleep(backoff)

        self._state = ConnectionState.STOPPED
        self._running = False
        await self._close()

    # ------------------------------------------------------------------
    async def stop(self) -> None:
        """Stop WebSocket client gracefully"""
        self.logger.info("🛑 Stopping WebSocket client...")
        self._stopping = True
        
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()
        
        await self._close()
        self._state = ConnectionState.STOPPED
        self.logger.info("✅ WebSocket client stopped")

    # ------------------------------------------------------------------
    async def subscribe(self, args: List[Dict[str, str]]) -> None:
        """Subscribe to channels"""
        self.subscriptions.extend(args)
        if not self.connected:
            self.logger.debug("Not connected, queued subscriptions for next connect")
            return
        await self._send({"op": "subscribe", "args": args})

    async def unsubscribe(self, args: List[Dict[str, str]]) -> None:
        """Unsubscribe from channels"""
        if not self.connected:
            self.logger.warning("Cannot unsubscribe, not connected")
            return
        await self._send({"op": "unsubscribe", "args": args})
    
    async def send_message(self, msg: Dict[str, Any]) -> None:
        """Send custom message"""
        if not self.connected:
            if len(self._message_queue) < 1000:
                self._message_queue.append(msg)
                self.logger.debug("Message queued for sending when reconnected")
            else:
                self.logger.warning("Message queue full, dropping message")
            return
        
        await self._send(msg)

    # ------------------------------------------------------------------
    async def _connect_and_run(self) -> None:
        """Main connection loop with ping/pong monitoring"""
        
        self.logger.info(f"🔌 Connecting to OKX WebSocket: {self.url}")

        async with websockets.connect(
            self.url,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            max_size=10_000_000
        ) as ws:

            self._ws = ws
            self._state = ConnectionState.CONNECTED
            self.logger.info("✅ WebSocket connected")
            
            self._last_message_ts = time.time()
            self._last_ping_ts = time.time()
            self._last_pong_ts = time.time()

            await self._subscribe_all()
            
            await self._flush_message_queue()
            
            self._ping_task = asyncio.create_task(self._ping_loop())
            self._watchdog_task = asyncio.create_task(self._connection_watchdog())

            async for raw in ws:
                self._last_message_ts = time.time()
                self._total_messages += 1

                if raw == "pong":
                    self._last_pong_ts = time.time()
                    self.logger.debug("🏓 Pong received")
                    continue

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self.logger.warning(f"⚠️ JSON decode failed: {raw[:100]!r}")
                    continue

                if "event" in data:
                    self._handle_event(data)
                    continue

                if self.on_message:
                    try:
                        await self.on_message(data)
                    except Exception as e:
                        self.logger.error(f"❌ Message handler error: {e}", exc_info=True)
            
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
            if self._watchdog_task and not self._watchdog_task.done():
                self._watchdog_task.cancel()

    # ------------------------------------------------------------------
    async def _subscribe_all(self) -> None:
        """Subscribe to all registered channels"""
        if not self.subscriptions:
            self.logger.info("No subscriptions to register")
            return
        
        self.logger.info(f"📡 Subscribing to {len(self.subscriptions)} channels...")
        await self._send({"op": "subscribe", "args": self.subscriptions})
    
    async def _flush_message_queue(self) -> None:
        """Send queued messages after reconnection"""
        if not self._message_queue:
            return
        
        self.logger.info(f"📤 Flushing {len(self._message_queue)} queued messages")
        while self._message_queue and self.connected:
            msg = self._message_queue.popleft()
            try:
                await self._send(msg)
            except Exception as e:
                self.logger.error(f"❌ Failed to send queued message: {e}")
    
    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive"""
        try:
            while self.connected and not self._stopping:
                await asyncio.sleep(self._ping_interval)
                
                if self.connected:
                    try:
                        await self._ws.ping()
                        self._last_ping_ts = time.time()
                        self.logger.debug("🏓 Ping sent")
                    except Exception as e:
                        self.logger.error(f"❌ Ping failed: {e}")
                        break
        except asyncio.CancelledError:
            self.logger.debug("Ping loop cancelled")
    
    async def _connection_watchdog(self) -> None:
        """Monitor connection health and force reconnect if dead"""
        try:
            while self.connected and not self._stopping:
                await asyncio.sleep(10)
                
                if not self.is_alive():
                    age = time.time() - self._last_message_ts
                    self.logger.error(
                        f"💀 Dead connection detected (no messages for {age:.1f}s)"
                    )
                    await self._close()
                    break
        except asyncio.CancelledError:
            self.logger.debug("Watchdog cancelled")

    # ------------------------------------------------------------------
    async def _send(self, msg: Dict[str, Any]) -> None:
        """Send message to WebSocket"""
        if not self._ws or self._ws.closed:
            self.logger.warning(f"⚠️ Cannot send, WS disconnected. Queueing: {msg.get('op', 'unknown')}")
            if len(self._message_queue) < 1000:
                self._message_queue.append(msg)
            return
        
        try:
            await self._ws.send(json.dumps(msg))
        except Exception as e:
            self.logger.error(f"❌ Send failed: {e}")
            raise

    # ------------------------------------------------------------------
    async def _close(self) -> None:
        """Close WebSocket connection"""
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
                self.logger.info("🔌 WebSocket closed")
            except Exception as e:
                self.logger.warning(f"⚠️ Close error: {e}")
        self._ws = None
        self._state = ConnectionState.DISCONNECTED

    # ------------------------------------------------------------------
    def _handle_event(self, data: Dict[str, Any]) -> None:
        """Handle WebSocket events"""
        
        event = data.get("event")
        arg = data.get("arg") or data.get("args")

        if event == "subscribe":
            self.logger.info(f"✅ Subscribed: {arg}")
        elif event == "unsubscribe":
            self.logger.info(f"❌ Unsubscribed: {arg}")
        elif event == "error":
            code = data.get("code")
            msg = data.get("msg")
            self.logger.error(f"⚠️ OKX Error: code={code}, msg={msg}")
        else:
            self.logger.debug(f"📡 WS Event: {data}")
