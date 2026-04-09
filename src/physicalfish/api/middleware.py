"""Simple token-bucket rate limiter middleware."""

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from physicalfish.logging_config import get_logger

logger = get_logger("rate_limiter")


class TokenBucket:
    """Token bucket for rate limiting."""

    def __init__(self, tokens_per_minute: int = 60):
        self.tokens_per_minute = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_update = time.time()
        self.max_tokens = tokens_per_minute

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time (tokens per minute)
        self.tokens = min(self.max_tokens, self.tokens + elapsed * (self.tokens_per_minute / 60.0))
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm."""

    def __init__(
        self,
        app,
        tokens_per_minute: int = 60,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.tokens_per_minute = tokens_per_minute
        self.exclude_paths = exclude_paths or ["/api/v1/health"]
        # In-memory storage: client_ip -> TokenBucket
        self._buckets: dict[str, TokenBucket] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        path = request.url.path

        # Skip rate limiting for excluded paths
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Get client IP
        client_ip = self._get_client_ip(request)

        # Get or create token bucket for this IP
        if client_ip not in self._buckets:
            self._buckets[client_ip] = TokenBucket(self.tokens_per_minute)
            logger.debug("new_bucket_created", client_ip=client_ip)

        bucket = self._buckets[client_ip]

        # Try to consume token
        if not bucket.consume(1):
            logger.warning("rate_limit_exceeded", client_ip=client_ip, path=path)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Rate limit exceeded. Try again later.",
                        "retry_after": int(60 / self.tokens_per_minute),
                    }
                },
            )

        # Process the request
        return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers (if behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"
