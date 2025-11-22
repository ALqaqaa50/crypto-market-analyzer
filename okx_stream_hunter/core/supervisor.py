# okx_stream_hunter/core/supervisor.py

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - psutil is optional
    psutil = None  # type: ignore


HealthCallback = Callable[[str], Awaitable[None]]


class StreamSupervisor:
    """
    Lightweight supervisor for monitoring stream health.

    - Monitors:
        * last market update timestamp
        * process memory / cpu (if psutil available)
    - Can trigger async callbacks when thresholds are violated.
    """

    def __init__(
        self,
        get_last_update_ts: Callable[[], float],
        logger: Optional[logging.Logger] = None,
        check_interval: float = 5.0,
        stale_after: float = 30.0,
        memory_warn_mb: float = 1500.0,
        cpu_warn_percent: float = 90.0,
        on_warning: Optional[HealthCallback] = None,
    ) -> None:
        self.get_last_update_ts = get_last_update_ts
        self.logger = logger or logging.getLogger(__name__)
        self.check_interval = check_interval
        self.stale_after = stale_after
        self.memory_warn_mb = memory_warn_mb
        self.cpu_warn_percent = cpu_warn_percent
        self.on_warning = on_warning

        self._running = False
        self._task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="stream-supervisor")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    async def _run(self) -> None:
        while self._running:
            try:
                await self._check_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Supervisor error: {e}", exc_info=True)

            await asyncio.sleep(self.check_interval)

    async def _check_once(self) -> None:
        now = time.time()
        last_ts = self.get_last_update_ts()
        delta = now - last_ts if last_ts else None

        if delta is None or delta > self.stale_after:
            msg = f"Market stream stale: last update {delta:.1f}s ago."
            self.logger.warning(msg)
            if self.on_warning:
                await self.on_warning(msg)

        if psutil is not None:
            p = psutil.Process()
            mem_mb = p.memory_info().rss / (1024 * 1024)
            cpu = p.cpu_percent(interval=0.0)

            if mem_mb > self.memory_warn_mb or cpu > self.cpu_warn_percent:
                msg = f"Resource warning: mem={mem_mb:.1f}MB, cpu={cpu:.1f}%"
                self.logger.warning(msg)
                if self.on_warning:
                    await self.on_warning(msg)
