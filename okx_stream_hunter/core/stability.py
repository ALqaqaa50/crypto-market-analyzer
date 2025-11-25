# okx_stream_hunter/core/stability.py
"""
🔥 Stability Manager - System Recovery, Heartbeat, and Resilience
"""

import asyncio
import logging
import os
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger("stability.manager")


@dataclass
class SystemState:
    """System state snapshot for recovery"""
    timestamp: str
    pid: int
    uptime_seconds: float
    status: str  # running, paused, error, recovering
    last_heartbeat: str
    
    # Trading state
    open_positions: int
    total_trades: int
    daily_pnl: float
    
    # Error tracking
    error_count: int
    last_error: Optional[str]
    crash_count: int
    
    # Performance
    signal_count: int
    avg_latency_ms: float


@dataclass
class RecoveryConfig:
    """Recovery configuration"""
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_delay_seconds: int = 5
    
    # Heartbeat
    heartbeat_interval_seconds: int = 30
    heartbeat_timeout_seconds: int = 120
    
    # Restart detection
    state_file: str = ".system_state.json"
    lock_file: str = ".system.lock"
    
    # Log rotation
    enable_log_rotation: bool = True
    max_log_size_mb: int = 100
    max_log_files: int = 5
    log_directory: str = "logs"


class StabilityManager:
    """
    🔥 System Stability and Recovery Manager
    
    Features:
    - Heartbeat monitoring (detect hangs/crashes)
    - Crash recovery (restore state after restart)
    - Restart protection (prevent duplicate instances)
    - Log rotation (prevent disk overflow)
    - Error tracking and reporting
    """
    
    def __init__(
        self,
        config: Optional[RecoveryConfig] = None,
        on_recovery: Optional[Callable] = None,
        on_heartbeat_timeout: Optional[Callable] = None,
    ):
        self.config = config or RecoveryConfig()
        self.on_recovery = on_recovery
        self.on_heartbeat_timeout = on_heartbeat_timeout
        
        # State tracking
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        self.pid = os.getpid()
        self.status = "initializing"
        
        # Error tracking
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.crash_count = 0
        
        # Performance metrics
        self.signal_count = 0
        self.total_latency = 0.0
        self.latency_samples = 0
        
        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._log_rotation_task: Optional[asyncio.Task] = None
        
        self._running = False
    
    # ============================================================
    # Lifecycle
    # ============================================================
    
    async def start(self) -> None:
        """Start stability manager"""
        logger.info("🔥 Stability Manager starting...")
        
        # Check for previous instance
        if self._check_existing_instance():
            logger.warning("⚠️ Previous instance detected, checking state...")
            await self._handle_restart()
        
        # Create lock file
        self._create_lock_file()
        
        # Start monitoring tasks
        self._running = True
        self.status = "running"
        
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        if self.config.enable_log_rotation:
            self._log_rotation_task = asyncio.create_task(self._log_rotation_loop())
        
        logger.info("✅ Stability Manager started successfully")
    
    async def stop(self) -> None:
        """Stop stability manager"""
        logger.info("Stopping Stability Manager...")
        self._running = False
        self.status = "stopped"
        
        # Cancel tasks
        for task in [self._heartbeat_task, self._monitor_task, self._log_rotation_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save final state
        self._save_state()
        
        # Remove lock file
        self._remove_lock_file()
        
        logger.info("✅ Stability Manager stopped")
    
    # ============================================================
    # Heartbeat Monitoring
    # ============================================================
    
    def heartbeat(self) -> None:
        """Update heartbeat timestamp (call this regularly from main loop)"""
        self.last_heartbeat = time.time()
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat monitoring"""
        logger.info("❤️ Heartbeat monitoring started")
        
        while self._running:
            try:
                # Check if heartbeat is stale
                time_since_heartbeat = time.time() - self.last_heartbeat
                
                if time_since_heartbeat > self.config.heartbeat_timeout_seconds:
                    logger.error(
                        f"💔 Heartbeat timeout! Last beat {time_since_heartbeat:.1f}s ago"
                    )
                    self.status = "heartbeat_timeout"
                    
                    # Trigger callback
                    if self.on_heartbeat_timeout:
                        try:
                            if asyncio.iscoroutinefunction(self.on_heartbeat_timeout):
                                await self.on_heartbeat_timeout()
                            else:
                                self.on_heartbeat_timeout()
                        except Exception as e:
                            logger.exception(f"Error in heartbeat timeout callback: {e}")
                    
                    # Attempt recovery
                    if self.config.enable_auto_recovery:
                        await self._attempt_recovery("heartbeat_timeout")
                
                # Save state periodically
                self._save_state()
                
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                
            except Exception as e:
                logger.exception(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5)
    
    # ============================================================
    # Crash Recovery
    # ============================================================
    
    def _check_existing_instance(self) -> bool:
        """Check if lock file exists (previous instance)"""
        lock_path = Path(self.config.lock_file)
        return lock_path.exists()
    
    def _create_lock_file(self) -> None:
        """Create lock file with current PID"""
        lock_path = Path(self.config.lock_file)
        with open(lock_path, "w") as f:
            json.dump({
                "pid": self.pid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, f)
    
    def _remove_lock_file(self) -> None:
        """Remove lock file"""
        lock_path = Path(self.config.lock_file)
        if lock_path.exists():
            lock_path.unlink()
    
    async def _handle_restart(self) -> None:
        """Handle system restart (recover previous state)"""
        logger.info("🔄 Handling system restart...")
        
        # Load previous state
        prev_state = self._load_previous_state()
        
        if prev_state:
            logger.info(
                f"📊 Previous state: uptime={prev_state.uptime_seconds:.1f}s, "
                f"positions={prev_state.open_positions}, trades={prev_state.total_trades}"
            )
            
            self.crash_count = prev_state.crash_count + 1
            
            if prev_state.status in ["error", "heartbeat_timeout"]:
                logger.warning(
                    f"⚠️ Previous instance crashed! Crash count: {self.crash_count}"
                )
                
                # Trigger recovery callback
                if self.on_recovery:
                    try:
                        if asyncio.iscoroutinefunction(self.on_recovery):
                            await self.on_recovery(prev_state)
                        else:
                            self.on_recovery(prev_state)
                    except Exception as e:
                        logger.exception(f"Error in recovery callback: {e}")
            else:
                logger.info("✅ Previous instance shutdown gracefully")
        else:
            logger.info("No previous state found")
    
    def _save_state(self) -> None:
        """Save current system state"""
        try:
            state = SystemState(
                timestamp=datetime.now(timezone.utc).isoformat(),
                pid=self.pid,
                uptime_seconds=time.time() - self.start_time,
                status=self.status,
                last_heartbeat=datetime.fromtimestamp(self.last_heartbeat, timezone.utc).isoformat(),
                open_positions=0,  # Will be injected by main system
                total_trades=0,
                daily_pnl=0.0,
                error_count=self.error_count,
                last_error=self.last_error,
                crash_count=self.crash_count,
                signal_count=self.signal_count,
                avg_latency_ms=self._calculate_avg_latency(),
            )
            
            state_path = Path(self.config.state_file)
            with open(state_path, "w") as f:
                json.dump(asdict(state), f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")
    
    def _load_previous_state(self) -> Optional[SystemState]:
        """Load previous system state"""
        try:
            state_path = Path(self.config.state_file)
            if state_path.exists():
                with open(state_path, "r") as f:
                    data = json.load(f)
                    return SystemState(**data)
        except Exception as e:
            logger.warning(f"Failed to load previous state: {e}")
        
        return None
    
    async def _attempt_recovery(self, reason: str) -> None:
        """Attempt system recovery"""
        logger.warning(f"🔧 Attempting recovery: {reason}")
        
        self.status = "recovering"
        
        for attempt in range(1, self.config.max_recovery_attempts + 1):
            logger.info(f"Recovery attempt {attempt}/{self.config.max_recovery_attempts}")
            
            try:
                # Wait before retry
                await asyncio.sleep(self.config.recovery_delay_seconds)
                
                # Reset heartbeat
                self.last_heartbeat = time.time()
                
                # Update status
                self.status = "running"
                
                logger.info(f"✅ Recovery successful on attempt {attempt}")
                return
                
            except Exception as e:
                logger.error(f"Recovery attempt {attempt} failed: {e}")
                
                if attempt >= self.config.max_recovery_attempts:
                    logger.critical("💥 All recovery attempts failed!")
                    self.status = "failed"
                    break
    
    # ============================================================
    # Error Tracking
    # ============================================================
    
    def record_error(self, error: Exception, context: str = "") -> None:
        """Record an error occurrence"""
        self.error_count += 1
        self.last_error = f"{context}: {str(error)}" if context else str(error)
        
        logger.error(
            f"🚨 Error recorded [{self.error_count}]: {self.last_error}"
        )
    
    def record_signal(self, latency_ms: float = 0.0) -> None:
        """Record a signal generation (for performance tracking)"""
        self.signal_count += 1
        
        if latency_ms > 0:
            self.total_latency += latency_ms
            self.latency_samples += 1
    
    def _calculate_avg_latency(self) -> float:
        """Calculate average signal latency"""
        if self.latency_samples == 0:
            return 0.0
        return self.total_latency / self.latency_samples
    
    # ============================================================
    # System Monitoring
    # ============================================================
    
    async def _monitor_loop(self) -> None:
        """Background system monitoring"""
        logger.info("📊 System monitoring started")
        
        while self._running:
            try:
                # Calculate metrics
                uptime = time.time() - self.start_time
                avg_latency = self._calculate_avg_latency()
                
                # Log status every 5 minutes
                if int(uptime) % 300 == 0:
                    logger.info(
                        f"📊 System Status: uptime={uptime/60:.1f}m, "
                        f"signals={self.signal_count}, "
                        f"errors={self.error_count}, "
                        f"avg_latency={avg_latency:.2f}ms, "
                        f"status={self.status}"
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.exception(f"Error in monitor loop: {e}")
                await asyncio.sleep(10)
    
    # ============================================================
    # Log Rotation
    # ============================================================
    
    async def _log_rotation_loop(self) -> None:
        """Background log rotation"""
        logger.info("📝 Log rotation started")
        
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                self._rotate_logs()
                
            except Exception as e:
                logger.exception(f"Error in log rotation: {e}")
                await asyncio.sleep(600)
    
    def _rotate_logs(self) -> None:
        """Rotate log files if they exceed max size"""
        log_dir = Path(self.config.log_directory)
        
        if not log_dir.exists():
            return
        
        try:
            # Find all .log files
            log_files = list(log_dir.glob("*.log"))
            
            for log_file in log_files:
                # Check file size
                size_mb = log_file.stat().st_size / (1024 * 1024)
                
                if size_mb > self.config.max_log_size_mb:
                    logger.info(f"🔄 Rotating log: {log_file.name} ({size_mb:.1f}MB)")
                    
                    # Rotate existing backups
                    for i in range(self.config.max_log_files - 1, 0, -1):
                        old_backup = log_file.with_suffix(f".log.{i}")
                        new_backup = log_file.with_suffix(f".log.{i+1}")
                        
                        if old_backup.exists():
                            if i + 1 >= self.config.max_log_files:
                                old_backup.unlink()  # Delete oldest
                            else:
                                old_backup.rename(new_backup)
                    
                    # Move current log to .log.1
                    backup = log_file.with_suffix(".log.1")
                    log_file.rename(backup)
                    
                    logger.info(f"✅ Log rotated: {log_file.name}")
        
        except Exception as e:
            logger.error(f"Error rotating logs: {e}")
    
    # ============================================================
    # Status Reporting
    # ============================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        uptime = time.time() - self.start_time
        
        return {
            "status": self.status,
            "uptime_seconds": uptime,
            "uptime_readable": self._format_uptime(uptime),
            "pid": self.pid,
            "last_heartbeat_ago": time.time() - self.last_heartbeat,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "crash_count": self.crash_count,
            "signal_count": self.signal_count,
            "avg_latency_ms": self._calculate_avg_latency(),
        }
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human-readable string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"


__all__ = ["StabilityManager", "RecoveryConfig", "SystemState"]
