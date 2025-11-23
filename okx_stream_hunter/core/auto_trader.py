from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .ai_brain import Signal  # نفس الـ TypedDict التي عندك
from ..integrations.trade_executor import TradeExecutor

logger = logging.getLogger("auto.trader")


class AutoTrader:
    """
    Ultra auto-trader controller.

    - Called whenever AI Brain emits a Signal
    - Forwards signals to TradeExecutor
    - Periodically calls executor.periodic_check
    - Provides hooks for future self-learning
    """

    def __init__(self, executor: Optional[TradeExecutor] = None) -> None:
        self.executor = executor or TradeExecutor()
        self._last_signal: Optional[Dict[str, Any]] = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        asyncio.create_task(self._watchdog_loop())

    async def stop(self) -> None:
        self._running = False
        await self.executor.close()

    async def handle_signal(self, sig: Signal) -> None:
        """
        Entry point from AI Brain.

        sig: TypedDict Signal from ai_brain.py
        """
        payload: Dict[str, Any] = {
            "dir": sig["direction"],
            "conf": float(sig["confidence"]),
            "price": float(sig["price"]),
            "reason": sig.get("reason", ""),
            "ts": sig.get("timestamp") or datetime.utcnow().isoformat(),
        }

        self._last_signal = payload
        logger.info(
            "AutoTrader received signal dir=%s conf=%.3f price=%.1f reason=%s",
            payload["dir"],
            payload["conf"],
            payload["price"],
            payload["reason"],
        )

        await self.executor.handle_signal(payload)

        # 🔁 HOOK: هنا بعدها مستقبلاً نضيف self-learning
        # مثال: تسجيل البيانات في ملف أو قاعدة بيانات:
        # await self.log_training_example(payload)

    async def _watchdog_loop(self) -> None:
        """Periodic timer to check stale positions etc."""
        while self._running:
            try:
                await self.executor.periodic_check()
            except Exception:
                logger.exception("Error in AutoTrader watchdog")
            await asyncio.sleep(30.0)
