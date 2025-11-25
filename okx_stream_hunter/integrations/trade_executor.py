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
from .position_manager import PositionManager, PositionManagerConfig

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
    🔥 Ultra auto-trader executor for OKX perpetual swaps.

    Features:
    - Receives AI signals (dir, conf, price, reason)
    - Decides whether to open/close/flip position
    - Sends signed REST orders to OKX
    - Integrates with PositionManager for TP/SL tracking
    - Supports automatic TP/SL orders via OKX API
    """

    def __init__(
        self,
        position_manager: Optional[PositionManager] = None,
    ) -> None:
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
        
        # 🔥 Position Manager integration
        self.position_manager = position_manager or PositionManager(
            config=PositionManagerConfig(),
            on_tp_hit=self._on_tp_hit,
            on_sl_hit=self._on_sl_hit,
            on_position_closed=self._on_position_closed,
        )

    # ========= HTTP utils =========

    async def connect(self) -> None:
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info("TradeExecutor HTTP session initialized")
        
        # Start position manager
        await self.position_manager.start()

    async def close(self) -> None:
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("TradeExecutor HTTP session closed")
        
        # Stop position manager
        await self.position_manager.stop()

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
    
    async def _place_tp_sl_orders(
        self,
        direction: Direction,
        size: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
    ) -> None:
        """
        Place TP and SL orders via OKX API.
        Uses algo orders (conditional orders) for TP/SL.
        """
        if not tp_price and not sl_price:
            return
        
        pos_side = "long" if direction == "long" else "short"
        
        # Place TP order (take-profit limit order)
        if tp_price:
            tp_side = "sell" if direction == "long" else "buy"
            body_tp = {
                "instId": self.cfg.symbol,
                "tdMode": self.cfg.td_mode,
                "side": tp_side,
                "posSide": pos_side,
                "ordType": "conditional",
                "sz": str(size),
                "triggerPx": str(tp_price),
                "orderPx": str(tp_price),
                "triggerPxType": "last",
                "reduceOnly": "true",
            }
            try:
                await self._request("POST", "/api/v5/trade/order-algo", body_tp)
                logger.info(f"✅ TP order placed @ {tp_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to place TP order: {e}")
        
        # Place SL order (stop-loss limit order)
        if sl_price:
            sl_side = "sell" if direction == "long" else "buy"
            body_sl = {
                "instId": self.cfg.symbol,
                "tdMode": self.cfg.td_mode,
                "side": sl_side,
                "posSide": pos_side,
                "ordType": "conditional",
                "sz": str(size),
                "triggerPx": str(sl_price),
                "orderPx": str(sl_price),
                "triggerPxType": "last",
                "reduceOnly": "true",
            }
            try:
                await self._request("POST", "/api/v5/trade/order-algo", body_sl)
                logger.info(f"✅ SL order placed @ {sl_price:.2f}")
            except Exception as e:
                logger.error(f"Failed to place SL order: {e}")

    # ========= Main signal handler =========

    async def handle_signal(self, signal: Dict[str, Any]) -> None:
        """
        🔥 Handle AI signal with full TP/SL support.
        
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
            
            # Open position in executor (sends market order)
            await self._open_position(direction, size)
            self.position.direction = direction
            self.position.size = size
            self.position.entry_price = price
            self.position.opened_ts = time.time()
            
            # Register position in Position Manager
            pm_position = self.position_manager.open_position(
                symbol=self.cfg.symbol,
                direction=direction,
                size=size,
                entry_price=price,
                reason=signal.get("reason", ""),
                confidence=conf,
            )
            
            # Place TP/SL orders via OKX
            await self._place_tp_sl_orders(
                direction=direction,
                size=size,
                tp_price=pm_position.tp_price,
                sl_price=pm_position.sl_price,
            )
            
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
            
            # Close old position
            await self._close_position()
            self.position_manager.close_position(
                symbol=self.cfg.symbol,
                close_price=price,
                reason="flip",
            )
            
            # Open new position
            size = self.cfg.base_size
            await self._open_position(direction, size)
            self.position.direction = direction
            self.position.size = size
            self.position.entry_price = price
            self.position.opened_ts = time.time()
            
            # Register new position
            pm_position = self.position_manager.open_position(
                symbol=self.cfg.symbol,
                direction=direction,
                size=size,
                entry_price=price,
                reason=f"flip_{signal.get('reason', '')}",
                confidence=conf,
            )
            
            # Place TP/SL orders
            await self._place_tp_sl_orders(
                direction=direction,
                size=size,
                tp_price=pm_position.tp_price,
                sl_price=pm_position.sl_price,
            )
        else:
            logger.debug(
                "Keeping existing position=%s, new_dir=%s, conf=%.3f",
                self.position.direction,
                direction,
                conf,
            )

    # ========= Periodic checks (time-based exit) =========

    async def periodic_check(self, current_price: Optional[float] = None) -> None:
        """
        Periodic position checks:
        - Update position with current price (for TP/SL/trailing)
        - Close stale positions after max_open_minutes
        """
        
        # Update position manager with current price
        if current_price and self.cfg.symbol in self.position_manager.positions:
            action = self.position_manager.update_position(
                symbol=self.cfg.symbol,
                current_price=current_price,
            )
            
            # If TP/SL hit, close via API
            if action and action.get("action") == "close":
                await self._close_position()
        
        # Check time-based exit (legacy support)
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
            self.position_manager.close_position(
                symbol=self.cfg.symbol,
                reason="max_age",
            )
    
    # ========= Position Manager Callbacks =========
    
    def _on_tp_hit(self, position, price: float) -> None:
        """Callback when TP is hit"""
        logger.info(f"🎯 TP HIT callback: {position.symbol} @ {price:.2f}")
    
    def _on_sl_hit(self, position, price: float) -> None:
        """Callback when SL is hit"""
        logger.warning(f"🛑 SL HIT callback: {position.symbol} @ {price:.2f}")
    
    def _on_position_closed(self, position, price: float) -> None:
        """Callback when position is closed"""
        logger.info(
            f"Position closed callback: {position.symbol} "
            f"PnL={position.realized_pnl:.4f} reason={position.close_reason}"
        )
