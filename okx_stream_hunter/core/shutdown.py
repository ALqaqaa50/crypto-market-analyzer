"""
Graceful Shutdown Handler
Ensures clean shutdown with data persistence
"""
import signal
import asyncio
from typing import Callable, List, Optional
from datetime import datetime, timezone

from ..utils.logger import get_logger


logger = get_logger(__name__)


class GracefulShutdown:
    """
    Handle graceful shutdown on SIGTERM, SIGINT, or manual trigger.
    
    Features:
    - Register cleanup callbacks
    - Timeout for cleanup
    - Force shutdown if cleanup takes too long
    - Final status logging
    """
    
    def __init__(self, timeout: int = 30):
        """
        Args:
            timeout: Maximum seconds to wait for cleanup
        """
        self.timeout = timeout
        self.is_shutting_down = False
        self.shutdown_initiated_at: Optional[datetime] = None
        self.cleanup_callbacks: List[Callable] = []
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        
        logger.info("Graceful shutdown handler initialized")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name
        logger.warning(f"\n{'='*60}")
        logger.warning(f"🛑 Shutdown signal received: {signal_name}")
        logger.warning(f"{'='*60}")
        
        if self.is_shutting_down:
            logger.warning("Shutdown already in progress, forcing exit...")
            self._force_shutdown()
            return
        
        self.is_shutting_down = True
        self.shutdown_initiated_at = datetime.now(timezone.utc)
    
    def register_cleanup(self, callback: Callable):
        """
        Register a cleanup callback.
        
        Callbacks should be async functions that return None.
        """
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    async def cleanup(self):
        """Execute all cleanup callbacks"""
        if not self.is_shutting_down:
            logger.warning("cleanup() called but shutdown not initiated")
            return
        
        logger.info(f"Starting cleanup ({len(self.cleanup_callbacks)} callbacks)...")
        
        start_time = asyncio.get_event_loop().time()
        
        for i, callback in enumerate(self.cleanup_callbacks, 1):
            try:
                logger.info(f"[{i}/{len(self.cleanup_callbacks)}] Running: {callback.__name__}")
                
                # Run with timeout
                await asyncio.wait_for(callback(), timeout=self.timeout / len(self.cleanup_callbacks))
                
                logger.info(f"✅ {callback.__name__} completed")
            
            except asyncio.TimeoutError:
                logger.error(f"⏱️  {callback.__name__} timed out")
            
            except Exception as e:
                logger.error(f"❌ {callback.__name__} failed: {e}")
        
        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"Cleanup completed in {elapsed:.2f}s")
        
        self._log_shutdown_summary()
    
    def _log_shutdown_summary(self):
        """Log final shutdown summary"""
        if self.shutdown_initiated_at:
            duration = (datetime.now(timezone.utc) - self.shutdown_initiated_at).total_seconds()
        else:
            duration = 0
        
        logger.info("="*60)
        logger.info("🛑 SHUTDOWN SUMMARY")
        logger.info("="*60)
        logger.info(f"Shutdown duration: {duration:.2f}s")
        logger.info(f"Cleanup callbacks executed: {len(self.cleanup_callbacks)}")
        logger.info(f"Status: {'⚠️  FORCED' if duration > self.timeout else '✅ CLEAN'}")
        logger.info("="*60)
        logger.info("👋 Goodbye!")
        logger.info("="*60)
    
    def _force_shutdown(self):
        """Force immediate shutdown"""
        logger.critical("⚠️  FORCING IMMEDIATE SHUTDOWN")
        import sys
        sys.exit(1)
    
    def should_shutdown(self) -> bool:
        """Check if shutdown was requested"""
        return self.is_shutting_down