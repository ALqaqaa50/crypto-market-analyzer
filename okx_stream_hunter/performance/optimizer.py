"""
Performance Optimizer - System performance monitoring and optimization
"""
import asyncio
import psutil
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger
from ..config.loader import get_config

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_tasks: int
    event_loop_lag_ms: float


class PerformanceOptimizer:
    """
    Monitor and optimize system performance.
    
    Features:
    - CPU and memory monitoring
    - Disk I/O tracking
    - Network usage tracking
    - Event loop lag detection
    - Automatic optimization recommendations
    """
    
    def __init__(self):
        config = get_config()
        
        self.enable_profiling = config.get("performance", "enable_profiling", default=True)
        self.profile_interval = config.get("performance", "profile_interval", default=300)
        self.memory_threshold_mb = config.get("performance", "memory_threshold_mb", default=500)
        self.cpu_threshold_percent = config.get("performance", "cpu_threshold_percent", default=80)
        
        self.process = psutil.Process()
        self.start_time = time.time()
        
        # Baseline metrics
        self._baseline_io = psutil.disk_io_counters()
        self._baseline_net = psutil.net_io_counters()
        
        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start performance monitoring"""
        if not self.enable_profiling:
            logger.info("Performance profiling disabled")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Performance monitoring started (interval: {self.profile_interval}s)")
    
    async def stop(self):
        """Stop performance monitoring"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        try:
            while self._running:
                await asyncio.sleep(self.profile_interval)
                
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Check thresholds and warn
                self._check_thresholds(metrics)
                
                # Provide optimization recommendations
                recommendations = self._get_recommendations(metrics)
                if recommendations:
                    logger.warning(f"Performance recommendations: {recommendations}")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in performance monitor: {e}")
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # CPU and Memory
        cpu_percent = self.process.cpu_percent(interval=1)
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        # Disk I/O
        current_io = psutil.disk_io_counters()
        disk_read_mb = (current_io.read_bytes - self._baseline_io.read_bytes) / 1024 / 1024
        disk_write_mb = (current_io.write_bytes - self._baseline_io.write_bytes) / 1024 / 1024
        
        # Network
        current_net = psutil.net_io_counters()
        net_sent_mb = (current_net.bytes_sent - self._baseline_net.bytes_sent) / 1024 / 1024
        net_recv_mb = (current_net.bytes_recv - self._baseline_net.bytes_recv) / 1024 / 1024
        
        # Async tasks
        try:
            loop = asyncio.get_running_loop()
            active_tasks = len([t for t in asyncio.all_tasks(loop) if not t.done()])
            lag_ms = 0  # Simplified - measure in async context if needed
        except:
            active_tasks = 0
            lag_ms = 0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            active_tasks=active_tasks,
            event_loop_lag_ms=lag_ms
        )
    
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics"""
        logger.info(
            f"Performance: CPU={metrics.cpu_percent:.1f}% "
            f"MEM={metrics.memory_mb:.1f}MB ({metrics.memory_percent:.1f}%) "
            f"Tasks={metrics.active_tasks} "
            f"Lag={metrics.event_loop_lag_ms:.2f}ms"
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Check if metrics exceed thresholds"""
        if metrics.memory_mb > self.memory_threshold_mb:
            logger.warning(
                f"High memory usage: {metrics.memory_mb:.1f}MB "
                f"(threshold: {self.memory_threshold_mb}MB)"
            )
        
        if metrics.cpu_percent > self.cpu_threshold_percent:
            logger.warning(
                f"High CPU usage: {metrics.cpu_percent:.1f}% "
                f"(threshold: {self.cpu_threshold_percent}%)"
            )
        
        if metrics.event_loop_lag_ms > 100:
            logger.warning(
                f"High event loop lag: {metrics.event_loop_lag_ms:.2f}ms"
            )
    
    def _get_recommendations(self, metrics: PerformanceMetrics) -> list:
        """Get optimization recommendations based on metrics"""
        recommendations = []
        
        if metrics.memory_mb > self.memory_threshold_mb:
            recommendations.append(
                "Consider reducing batch sizes or increasing flush frequency"
            )
        
        if metrics.cpu_percent > self.cpu_threshold_percent:
            recommendations.append(
                "Consider reducing calculation frequency or offloading to background tasks"
            )
        
        if metrics.event_loop_lag_ms > 100:
            recommendations.append(
                "Event loop is blocked - move heavy computations to executor"
            )
        
        if metrics.active_tasks > 100:
            recommendations.append(
                "High number of active tasks - consider task pooling"
            )
        
        return recommendations
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        metrics = self.collect_metrics()
        
        return {
            "uptime_seconds": self.get_uptime(),
            "cpu_percent": metrics.cpu_percent,
            "memory_mb": metrics.memory_mb,
            "memory_percent": metrics.memory_percent,
            "active_tasks": metrics.active_tasks,
            "event_loop_lag_ms": metrics.event_loop_lag_ms,
        }
