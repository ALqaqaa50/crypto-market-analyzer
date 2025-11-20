"""
Rate Limiter for Webhooks and API Calls
Prevents overwhelming external services
"""
import asyncio
import time
from typing import Dict, Optional
from collections import deque

from ..utils.logger import get_logger


logger = get_logger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter.
    
    Allows bursts but enforces average rate.
    """
    
    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens per second
            capacity: Bucket capacity (max burst)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens.
        
        Returns:
            True if tokens acquired, False otherwise
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            
            # Refill tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, tokens: float = 1.0):
        """Wait until token is available"""
        while not await self.acquire(tokens):
            await asyncio.sleep(0.01)


class RateLimiter:
    """
    Multi-endpoint rate limiter with different limits per target.
    """
    
    def __init__(self):
        self.limiters: Dict[str, TokenBucket] = {}
        self.stats: Dict[str, Dict] = {}
    
    def add_limit(
        self,
        name: str,
        requests_per_second: float,
        burst: Optional[float] = None
    ):
        """
        Add rate limit for a named endpoint.
        
        Args:
            name: Endpoint identifier
            requests_per_second: Rate limit
            burst: Burst capacity (defaults to rate * 2)
        """
        if burst is None:
            burst = requests_per_second * 2
        
        self.limiters[name] = TokenBucket(requests_per_second, burst)
        self.stats[name] = {
            'total_requests': 0,
            'throttled_requests': 0,
            'total_wait_time': 0.0,
        }
        
        logger.info(
            f"Rate limiter added: {name} "
            f"({requests_per_second} req/s, burst={burst})"
        )
    
    async def acquire(self, name: str, tokens: float = 1.0) -> bool:
        """
        Try to acquire permission for request.
        
        Returns:
            True if allowed, False if rate limited
        """
        if name not in self.limiters:
            logger.warning(f"No rate limiter for '{name}', allowing request")
            return True
        
        limiter = self.limiters[name]
        acquired = await limiter.acquire(tokens)
        
        self.stats[name]['total_requests'] += 1
        
        if not acquired:
            self.stats[name]['throttled_requests'] += 1
        
        return acquired
    
    async def wait(self, name: str, tokens: float = 1.0):
        """Wait for permission (blocks until allowed)"""
        if name not in self.limiters:
            return
        
        start = time.monotonic()
        await self.limiters[name].wait_for_token(tokens)
        wait_time = time.monotonic() - start
        
        self.stats[name]['total_wait_time'] += wait_time
        
        if wait_time > 0.1:
            logger.debug(f"Rate limited {name}: waited {wait_time:.3f}s")
    
    def get_stats(self, name: str) -> Dict:
        """Get statistics for an endpoint"""
        return self.stats.get(name, {})
    
    def get_all_stats(self) -> Dict:
        """Get all statistics"""
        return self.stats.copy()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter (singleton)"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter