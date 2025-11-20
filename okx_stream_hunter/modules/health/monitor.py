"""
Health monitoring and heartbeat system
"""
import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional
import time

try:
    import aiohttp
except ImportError:
    aiohttp = None

from ...utils.logger import get_logger
from ...config.loader import get_config


logger = get_logger(__name__)


class HealthMonitor:
    """Monitor system health and send periodic heartbeats"""
    
    def __init__(self):
        config = get_config()
        
        self.enabled = config.get("health", "enabled", default=True)
        self.heartbeat_interval = config.get("health", "heartbeat_interval", default=60)
        self.heartbeat_url = config.get("webhooks", "heartbeat_url", default="")
        
        # Metrics
        self.start_time = time.time()
        self.last_tick_time: Optional[datetime] = None
        self.total_ticks = 0
        self.total_errors = 0
        self.total_candles = 0
        self.total_db_writes = 0
        
        # Stream status
        self.streams_status: Dict[str, bool] = {
            "trades": False,
            "orderbook": False,
            "funding": False,
            "open_interest": False,
            "liquidations": False,
        }
        
        # Background task
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    def record_tick(self):
        """Record a new tick received"""
        self.last_tick_time = datetime.now(timezone.utc)
        self.total_ticks += 1
        self.streams_status["trades"] = True
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error"""
        self.total_errors += 1
        logger.error(f"Error recorded: {error_type}")
    
    def record_candle(self):
        """Record a candle generated"""
        self.total_candles += 1
    
    def record_db_write(self):
        """Record a database write"""
        self.total_db_writes += 1
    
    def update_stream_status(self, stream: str, active: bool):
        """Update status of a specific stream"""
        if stream in self.streams_status:
            self.streams_status[stream] = active
    
    def get_status(self) -> Dict:
        """Get current health status"""
        uptime = time.time() - self.start_time
        
        # Calculate tick age
        tick_age = None
        if self.last_tick_time:
            tick_age = (datetime.now(timezone.utc) - self.last_tick_time).total_seconds()
        
        # Determine health status
        is_healthy = True
        health_issues = []
        
        if tick_age and tick_age > 30:
            is_healthy = False
            health_issues.append(f"No ticks for {tick_age:.0f}s")
        
        if not any(self.streams_status.values()):
            is_healthy = False
            health_issues.append("All streams inactive")
        
        error_rate = self.total_errors / max(self.total_ticks, 1)
        if error_rate > 0.01:  # 1% error rate
            health_issues.append(f"High error rate: {error_rate*100:.2f}%")
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "issues": health_issues,
            "uptime_seconds": int(uptime),
            "uptime_human": self._format_uptime(uptime),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "total_ticks": self.total_ticks,
                "total_candles": self.total_candles,
                "total_errors": self.total_errors,
                "total_db_writes": self.total_db_writes,
                "error_rate": error_rate,
                "last_tick_age_seconds": tick_age,
            },
            "streams": self.streams_status,
        }
    
    async def send_heartbeat(self):
        """Send heartbeat to configured webhook"""
        if not self.heartbeat_url or not aiohttp:
            return
        
        status = self.get_status()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.heartbeat_url,
                    json=status,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status >= 300:
                        logger.warning(
                            f"Heartbeat webhook returned {resp.status}"
                        )
        except Exception as e:
            logger.warning(f"Failed to send heartbeat: {e}")
    
    async def start_heartbeat_loop(self):
        """Start background heartbeat loop"""
        if not self.enabled:
            logger.info("Health monitoring disabled")
            return
        
        logger.info(
            f"Starting heartbeat loop (interval: {self.heartbeat_interval}s)"
        )
        
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self.send_heartbeat()
            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
    
    def start(self):
        """Start health monitoring"""
        if self.enabled and not self._heartbeat_task:
            self._heartbeat_task = asyncio.create_task(
                self.start_heartbeat_loop()
            )
    
    async def stop(self):
        """Stop health monitoring"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
    
    @staticmethod
    def _format_uptime(seconds: float) -> str:
        """Format uptime in human-readable form"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        
        return " ".join(parts)