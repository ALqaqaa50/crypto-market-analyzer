"""
Adaptive Rate Limiter - PHASE 3
Smart throttling with burst protection and adaptive limits
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class AdaptiveRateLimiter:
    """Adaptive rate limiter with intelligent throttling"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        self.request_history = deque(maxlen=1000)
        self.error_history = deque(maxlen=100)
        
        self.base_limit = config.get('base_requests_per_second', 10)
        self.current_limit = self.base_limit
        self.min_limit = config.get('min_requests_per_second', 1)
        self.max_limit = config.get('max_requests_per_second', 50)
        
        self.burst_size = config.get('burst_size', 20)
        self.burst_window_seconds = config.get('burst_window_seconds', 10)
        
        self.adaptation_rate = config.get('adaptation_rate', 0.1)
        self.recovery_rate = config.get('recovery_rate', 0.05)
        
        self.error_threshold = config.get('error_threshold', 0.1)
        self.error_backoff_factor = config.get('error_backoff_factor', 0.5)
        
        self.last_adaptation = datetime.now()
        self.adaptation_interval = timedelta(seconds=30)
        
        self.stats = {
            'total_requests': 0,
            'throttled_requests': 0,
            'burst_detected': 0,
            'adaptations': 0
        }
        
        logger.info(f"⚡ Adaptive Rate Limiter initialized: {self.base_limit} req/s")
    
    async def acquire(self, resource: str = 'default') -> bool:
        """Acquire permission for request"""
        now = datetime.now()
        
        if not await self._check_rate_limit(now):
            self.stats['throttled_requests'] += 1
            logger.debug(f"⏸️ Rate limit: {resource}")
            return False
        
        if not await self._check_burst_limit(now):
            self.stats['throttled_requests'] += 1
            self.stats['burst_detected'] += 1
            logger.warning(f"🚨 Burst detected: {resource}")
            return False
        
        self.request_history.append({
            'timestamp': now,
            'resource': resource,
            'approved': True
        })
        
        self.stats['total_requests'] += 1
        
        await self._adapt_limits(now)
        
        return True
    
    async def record_error(self, error_type: str = 'unknown'):
        """Record error for adaptive throttling"""
        self.error_history.append({
            'timestamp': datetime.now(),
            'type': error_type
        })
        
        await self._adjust_for_errors()
    
    async def record_success(self):
        """Record successful request for rate increase"""
        if len(self.request_history) >= 10:
            recent_requests = list(self.request_history)[-10:]
            all_successful = all(r.get('approved', False) for r in recent_requests)
            
            if all_successful and self.current_limit < self.max_limit:
                self.current_limit = min(
                    self.current_limit * (1 + self.recovery_rate),
                    self.max_limit
                )
                logger.debug(f"📈 Rate limit increased: {self.current_limit:.2f} req/s")
    
    async def _check_rate_limit(self, now: datetime) -> bool:
        """Check if within rate limit"""
        window_start = now - timedelta(seconds=1)
        
        recent_requests = [
            r for r in self.request_history
            if r['timestamp'] > window_start
        ]
        
        return len(recent_requests) < self.current_limit
    
    async def _check_burst_limit(self, now: datetime) -> bool:
        """Check if burst limit exceeded"""
        window_start = now - timedelta(seconds=self.burst_window_seconds)
        
        recent_requests = [
            r for r in self.request_history
            if r['timestamp'] > window_start
        ]
        
        return len(recent_requests) < self.burst_size
    
    async def _adapt_limits(self, now: datetime):
        """Adapt rate limits based on performance"""
        if now - self.last_adaptation < self.adaptation_interval:
            return
        
        self.last_adaptation = now
        
        window_start = now - timedelta(seconds=60)
        recent_requests = [
            r for r in self.request_history
            if r['timestamp'] > window_start
        ]
        
        if not recent_requests:
            return
        
        recent_errors = [
            e for e in self.error_history
            if e['timestamp'] > window_start
        ]
        
        error_rate = len(recent_errors) / len(recent_requests) if recent_requests else 0
        
        if error_rate > self.error_threshold:
            await self._adjust_for_errors()
        else:
            if self.current_limit < self.base_limit:
                self.current_limit = min(
                    self.current_limit * (1 + self.recovery_rate),
                    self.base_limit
                )
                logger.info(f"✅ Rate limit recovering: {self.current_limit:.2f} req/s")
        
        self.stats['adaptations'] += 1
    
    async def _adjust_for_errors(self):
        """Reduce rate limit due to errors"""
        old_limit = self.current_limit
        self.current_limit = max(
            self.current_limit * self.error_backoff_factor,
            self.min_limit
        )
        
        if old_limit != self.current_limit:
            logger.warning(f"⬇️ Rate limit reduced: {old_limit:.2f} → {self.current_limit:.2f} req/s")
    
    def get_current_rate(self) -> float:
        """Get current rate limit"""
        return self.current_limit
    
    def get_stats(self) -> Dict:
        """Get limiter statistics"""
        now = datetime.now()
        window_start = now - timedelta(seconds=60)
        
        recent_requests = [
            r for r in self.request_history
            if r['timestamp'] > window_start
        ]
        
        recent_errors = [
            e for e in self.error_history
            if e['timestamp'] > window_start
        ]
        
        return {
            **self.stats,
            'current_limit': self.current_limit,
            'requests_last_minute': len(recent_requests),
            'errors_last_minute': len(recent_errors),
            'error_rate': len(recent_errors) / len(recent_requests) if recent_requests else 0,
            'utilization': len(recent_requests) / (self.current_limit * 60) if self.current_limit > 0 else 0
        }
    
    def reset(self):
        """Reset to base limits"""
        self.current_limit = self.base_limit
        logger.info(f"🔄 Rate limiter reset to base: {self.base_limit} req/s")


class AdaptiveThrottler:
    """Adaptive throttling decorator"""
    
    def __init__(self, limiter: AdaptiveRateLimiter, resource: str = 'default'):
        self.limiter = limiter
        self.resource = resource
    
    async def __call__(self, func):
        """Throttle function execution"""
        async def wrapper(*args, **kwargs):
            max_retries = 5
            retry_delay = 0.1
            
            for attempt in range(max_retries):
                if await self.limiter.acquire(self.resource):
                    try:
                        result = await func(*args, **kwargs)
                        await self.limiter.record_success()
                        return result
                    except Exception as e:
                        await self.limiter.record_error(type(e).__name__)
                        raise
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
            
            raise Exception(f"Rate limit exceeded for {self.resource}")
        
        return wrapper
