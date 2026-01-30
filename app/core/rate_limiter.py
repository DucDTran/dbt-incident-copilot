"""
Rate Limiter - Token bucket rate limiting for API calls.

This module provides rate limiting functionality to prevent exceeding
API quotas for Gemini, BigQuery, and embedding services.

Features:
- Token bucket algorithm with smooth rate limiting
- Per-service rate limits (RPM, TPM, QPM)
- Async-compatible with proper locking
- Automatic backoff when limits are reached

Usage:
    from app.core.rate_limiter import RateLimiter, rate_limit
    
    # Using the decorator
    @rate_limit("gemini")
    async def call_gemini_api():
        ...
    
    # Using the limiter directly
    limiter = RateLimiter.get_instance()
    await limiter.acquire("gemini", tokens=100)
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Callable, Optional, TypeVar, ParamSpec
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class RateLimitService(str, Enum):
    """Supported services for rate limiting."""
    GEMINI = "gemini"
    BIGQUERY = "bigquery"
    EMBEDDING = "embedding"


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.
    
    Implements the token bucket algorithm where tokens are added at a 
    constant rate up to a maximum capacity. Each request consumes tokens.
    
    Attributes:
        capacity: Maximum number of tokens in the bucket
        refill_rate: Tokens added per second
        tokens: Current number of available tokens
        last_refill: Timestamp of last token refill
    """
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = field(default=0.0)
    last_refill: float = field(default_factory=time.monotonic)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize with full bucket."""
        self.tokens = self.capacity
    
    async def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens (None = wait indefinitely)
            
        Returns:
            True if tokens were acquired, False if timeout reached
        """
        start_time = time.monotonic()
        
        async with self._lock:
            while True:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        return False
                    wait_time = min(wait_time, remaining)
                
                # Release lock while waiting
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate, no lock)."""
        elapsed = time.monotonic() - self.last_refill
        return min(self.capacity, self.tokens + elapsed * self.refill_rate)


class RateLimiter:
    """
    Centralized rate limiter for all API services.
    
    Manages separate token buckets for each service with configurable
    limits from settings.
    
    Example:
        limiter = RateLimiter.get_instance()
        
        # Acquire for Gemini API call
        await limiter.acquire("gemini")
        
        # Acquire with token count (for TPM limiting)
        await limiter.acquire("gemini", tokens=500)
        
        # Check if rate limited
        if limiter.is_limited("bigquery"):
            logger.warning("BigQuery rate limited")
    """
    
    _instance: Optional["RateLimiter"] = None
    
    def __init__(self) -> None:
        """Initialize rate limiter with buckets from settings."""
        self._settings = get_settings()
        self._buckets: dict[str, TokenBucket] = {}
        self._enabled = self._settings.rate_limits.enabled
        
        if self._enabled:
            self._init_buckets()
    
    def _init_buckets(self) -> None:
        """Initialize token buckets from settings."""
        limits = self._settings.rate_limits
        
        # Gemini RPM bucket
        self._buckets["gemini_rpm"] = TokenBucket(
            capacity=limits.gemini_rpm,
            refill_rate=limits.gemini_rpm / 60.0,  # per second
        )
        
        # Gemini TPM bucket (for token-based limiting)
        self._buckets["gemini_tpm"] = TokenBucket(
            capacity=limits.gemini_tpm,
            refill_rate=limits.gemini_tpm / 60.0,
        )
        
        # BigQuery QPM bucket
        self._buckets["bigquery"] = TokenBucket(
            capacity=limits.bigquery_qpm,
            refill_rate=limits.bigquery_qpm / 60.0,
        )
        
        # Embedding RPM bucket
        self._buckets["embedding"] = TokenBucket(
            capacity=limits.embedding_rpm,
            refill_rate=limits.embedding_rpm / 60.0,
        )
        
        logger.info(
            "Rate limiter initialized",
            extra={
                "gemini_rpm": limits.gemini_rpm,
                "gemini_tpm": limits.gemini_tpm,
                "bigquery_qpm": limits.bigquery_qpm,
                "embedding_rpm": limits.embedding_rpm,
            }
        )
    
    @classmethod
    def get_instance(cls) -> "RateLimiter":
        """Get or create the singleton rate limiter instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None
    
    async def acquire(
        self,
        service: str | RateLimitService,
        tokens: float = 1.0,
        timeout: Optional[float] = 30.0,
    ) -> bool:
        """
        Acquire rate limit tokens for a service.
        
        Args:
            service: Service name ("gemini", "bigquery", "embedding")
            tokens: Number of tokens to acquire (1 for request counting,
                   more for token-based limits)
            timeout: Maximum wait time in seconds
            
        Returns:
            True if tokens acquired, False if timeout or disabled
        """
        if not self._enabled:
            return True
        
        service_name = service.value if isinstance(service, RateLimitService) else service
        
        # For Gemini, we need to acquire from both RPM and TPM buckets
        if service_name == "gemini":
            rpm_result = await self._buckets["gemini_rpm"].acquire(1.0, timeout)
            if not rpm_result:
                logger.warning("Gemini RPM rate limit reached")
                return False
            
            if tokens > 1:
                tpm_result = await self._buckets["gemini_tpm"].acquire(tokens, timeout)
                if not tpm_result:
                    logger.warning(f"Gemini TPM rate limit reached (requested {tokens} tokens)")
                    return False
            return True
        
        bucket = self._buckets.get(service_name)
        if bucket is None:
            logger.warning(f"Unknown rate limit service: {service_name}")
            return True
        
        result = await bucket.acquire(tokens, timeout)
        if not result:
            logger.warning(f"Rate limit reached for {service_name}")
        return result
    
    def is_limited(self, service: str | RateLimitService) -> bool:
        """
        Check if a service is currently rate limited.
        
        Args:
            service: Service name to check
            
        Returns:
            True if rate limited (no tokens available), False otherwise
        """
        if not self._enabled:
            return False
        
        service_name = service.value if isinstance(service, RateLimitService) else service
        
        if service_name == "gemini":
            rpm_bucket = self._buckets.get("gemini_rpm")
            return rpm_bucket is not None and rpm_bucket.available_tokens < 1
        
        bucket = self._buckets.get(service_name)
        return bucket is not None and bucket.available_tokens < 1
    
    def get_stats(self) -> dict[str, dict[str, float]]:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dict mapping service names to their stats (available_tokens, capacity)
        """
        if not self._enabled:
            return {"enabled": False}
        
        return {
            name: {
                "available_tokens": bucket.available_tokens,
                "capacity": bucket.capacity,
                "utilization": 1 - (bucket.available_tokens / bucket.capacity),
            }
            for name, bucket in self._buckets.items()
        }


def rate_limit(
    service: str | RateLimitService,
    tokens: float = 1.0,
    timeout: Optional[float] = 30.0,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to apply rate limiting to a function.
    
    Args:
        service: Service name for rate limiting
        tokens: Tokens to acquire per call
        timeout: Maximum wait time
        
    Returns:
        Decorated function with rate limiting
        
    Example:
        @rate_limit("gemini")
        async def call_gemini():
            ...
        
        @rate_limit("gemini", tokens=500)  # For token-heavy calls
        async def call_gemini_with_tokens():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            limiter = RateLimiter.get_instance()
            acquired = await limiter.acquire(service, tokens, timeout)
            if not acquired:
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {service}, timeout after {timeout}s"
                )
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # For sync functions, run in event loop
            limiter = RateLimiter.get_instance()
            loop = asyncio.get_event_loop()
            acquired = loop.run_until_complete(limiter.acquire(service, tokens, timeout))
            if not acquired:
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {service}, timeout after {timeout}s"
                )
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded and timeout reached."""
    pass
