from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any

import aiohttp

from ..config.loader import get_settings

logger = logging.getLogger("trade.executor")

Direction = Literal["long", "short", "flat"]


@dataclass
class TradeExecutorConfig:
    symbol: str
    td_mode: str
    leverage: int
    base_size: float
    min_conf: float
    flip_conf: float
    max_open_minutes: int


@dataclass
class PositionState:
    direction: Direction = "flat"
    size: float = 0.0
    entry_price: Optional[float] = None
    opened_ts: Optional[float] = None
    last_signal_ts: Optional[float] = None
    last_confidence: float = 0.0


class TradeExecutor:
    """
    Ultra auto-trader executor for OKX perpetual swaps.

    - Receives AI signals (dir, conf, price, reason)
    - Decides whether to open/close/flip position
    - Sends signed REST orders to OKX
    """

    def __init__(self) -> None:
        cfg = get_settings()
        okx_cfg = cfg.okx
        trade_cfg = cfg.trading

        self.api_key: str = okx_cfg.api_key
        self.secret_key: str = okx_cfg.secret_key
        self.passphrase: str = okx_cfg.passphrase
        self.base_url: str = (
            "https://www.okx.com" if not okx_cfg.sandbox else "https://www.okx.com"
        )

        self.cfg = TradeExecutorConfig(
            symbol=trade_cfg.symbol,
            td_mode=trade_cfg.td_mode,
            leverage=trade_cfg.leverage,
            base_size=float(trade_cfg.base_size),
            min_conf=float(trade_cfg.min_conf),
            flip_conf=float(trade_cfg.flip_conf),
            max_open_minutes=int(trade_cfg.max_open_minutes),
        )

        self.session: Optional[aiohttp.ClientSession] = None
        self.position = PositionState()

    # ========= HTTP utils =========

    async def connect(self) -> None:
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("TradeExecutor HTTP session initialized")

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("TradeExecutor HTTP session closed")

    def _sign(self, ts: str, method: str, path: str, body: str) -> str:
        msg = f"{ts}{method}{path}{body}"
        mac = hmac.new(
            self.secret_key.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode()

    async def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        assert self.session is not None, "TradeExecutor not connected"

        ts = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        body_json = json.dumps(body) if body else ""
        sign = self._sign(ts, method, path, body_json)

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

        url = self.base_url + path
        async with self.session.request(
            method, url, headers=headers, data=body_json or None
        ) as resp:
            text = await resp.text()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.error("Non-JSON response from OKX: %s", text)
                raise

            if data.get("code") != "0":
                logger.error("OKX error %s: %s", data.get("code"), data.get("msg"))
            else:
                logger.debug("OKX response: %s", data)

            return data

    # ========= Order helpers =========

    async def _place_order(
        self,
        side: Literal["buy", "sell"],
        pos_side: Literal["long", "short"],
        size: float,
        reduce_only: bool = False,
    ) -> None:
        body = {
            "instId": self.cfg.symbol,
            "tdMode": self.cfg.td_mode,
            "side": side,
            "posSide": pos_side,
            "ordType": "market",
            "sz": str(size),
        }
        if reduce_only:
            body["reduceOnly"] = "true"

        logger.info(
            "Placing order: side=%s posSide=%s sz=%s reduceOnly=%s",
            side,
            pos_side,
            size,
            reduce_only,
        )
        await self._request("POST", "/api/v5/trade/order", body)

    async def _open_position(self, direction: Direction, size: float) -> None:
        if direction == "long":
            await self._place_order("buy", "long", size, reduce_only=False)
        elif direction == "short":
            await self._place_order("sell", "short", size, reduce_only=False)

    async def _close_position(self) -> None:
        if self.position.direction == "long":
            await self._place_order(
                "sell", "long", self.position.size, reduce_only=True
            )
        elif self.position.direction == "short":
            await self._place_order(
                "buy", "short", self.position.size, reduce_only=True
            )

        logger.info("Position closed")
        self.position = PositionState()

    # ========= Main signal handler =========

    async def handle_signal(self, signal: Dict[str, Any]) -> None:
        """
        signal = {
          "dir": "long" | "short" | "flat",
          "conf": 0.0-1.0,
          "price": float,
          "reason": str,
          "ts": datetime or iso string
        }
        """
        await self.connect()

        direction: Direction = signal.get("dir", "flat")  # type: ignore
        conf: float = float(signal.get("conf", 0.0))
        price: float = float(signal.get("price", 0.0))

        self.position.last_signal_ts = time.time()
        self.position.last_confidence = conf

        if direction == "flat":
            logger.debug("Received flat signal, no action")
            return

        if conf < self.cfg.min_conf:
            logger.info("Signal conf %.3f < min_conf %.3f -> ignore", conf, self.cfg.min_conf)
            return

        # --- 1) No position: open new one ---
        if self.position.direction == "flat":
            size = self.cfg.base_size
            logger.info(
                "Opening new %s position, size=%s, conf=%.3f price=%.1f reason=%s",
                direction,
                size,
                conf,
                price,
                signal.get("reason"),
            )
            await self._open_position(direction, size)
            self.position.direction = direction
            self.position.size = size
            self.position.entry_price = price
            self.position.opened_ts = time.time()
            return

        # --- 2) Already in position: maybe flip or hold ---
        if direction != self.position.direction and conf >= self.cfg.flip_conf:
            logger.info(
                "Flipping position %s -> %s (conf=%.3f >= flip_conf=%.3f)",
                self.position.direction,
                direction,
                conf,
                self.cfg.flip_conf,
            )
            await self._close_position()
            size = self.cfg.base_size
            await self._open_position(direction, size)
            self.position.direction = direction
            self.position.size = size
            self.position.entry_price = price
            self.position.opened_ts = time.time()
        else:
            logger.debug(
                "Keeping existing position=%s, new_dir=%s, conf=%.3f",
                self.position.direction,
                direction,
                conf,
            )

    # ========= Periodic checks (time-based exit) =========

    async def periodic_check(self, current_price: Optional[float] = None) -> None:
        """Close stale positions after max_open_minutes."""
        if self.position.direction == "flat":
            return

        if not self.position.opened_ts:
            return

        age_min = (time.time() - self.position.opened_ts) / 60.0
        if age_min >= self.cfg.max_open_minutes:
            logger.info(
                "Position age %.1f min >= max_open_minutes=%d -> closing",
                age_min,
                self.cfg.max_open_minutes,
            )
            await self._close_position()
