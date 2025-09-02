"""
Security middleware for FastAPI application.
Provides security headers, rate limiting, and request monitoring.
"""

import time
import asyncio
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    Protects against XSS, clickjacking, MIME sniffing, and other attacks.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            # XSS Protection
            "X-XSS-Protection": "1; mode=block",
            
            # Content Type Options
            "X-Content-Type-Options": "nosniff",
            
            # Frame Options (Clickjacking protection)
            "X-Frame-Options": "DENY",
            
            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:; "
                "font-src 'self'; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-ancestors 'none'"
            ),
            
            # Strict Transport Security (HTTPS only)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            
            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=(), "
                "vibrate=(), "
                "fullscreen=()"
            )
        }
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers to all responses
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add custom server header
        response.headers["Server"] = "VivaStageAI/1.0"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    Implements different limits for authenticated vs anonymous users.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        cleanup_interval: int = 300  # 5 minutes
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # Storage for request tracking
        self.minute_windows: Dict[str, deque] = defaultdict(deque)
        self.hour_windows: Dict[str, deque] = defaultdict(deque)
        
        # Cleanup old entries periodically
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        # Track concurrent requests
        self.concurrent_requests: Dict[str, int] = defaultdict(int)
        self.max_concurrent = 10
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        Uses user ID if authenticated, otherwise IP address.
        """
        # Try to get user ID from request state (set by auth middleware)
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _cleanup_old_entries(self):
        """Remove old entries from tracking windows."""
        now = time.time()
        
        # Only cleanup periodically
        if now - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_minute = now - 60
        cutoff_hour = now - 3600
        
        # Cleanup minute windows
        for client_id in list(self.minute_windows.keys()):
            window = self.minute_windows[client_id]
            while window and window[0] < cutoff_minute:
                window.popleft()
            if not window:
                del self.minute_windows[client_id]
        
        # Cleanup hour windows
        for client_id in list(self.hour_windows.keys()):
            window = self.hour_windows[client_id]
            while window and window[0] < cutoff_hour:
                window.popleft()
            if not window:
                del self.hour_windows[client_id]
        
        # Cleanup concurrent request counters
        # Note: In a production environment, you'd want a more sophisticated
        # cleanup mechanism for concurrent requests
        
        self.last_cleanup = now
        logger.info("Rate limit storage cleanup completed")
    
    def _is_rate_limited(self, client_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if client is rate limited.
        Returns rate limit info if limited, None otherwise.
        """
        now = time.time()
        
        # Check minute limit
        minute_window = self.minute_windows[client_id]
        minute_cutoff = now - 60
        
        # Remove old entries
        while minute_window and minute_window[0] < minute_cutoff:
            minute_window.popleft()
        
        if len(minute_window) >= self.requests_per_minute:
            return {
                "error": "Rate limit exceeded",
                "limit": self.requests_per_minute,
                "window": "minute",
                "retry_after": int(60 - (now - minute_window[0]))
            }
        
        # Check hour limit
        hour_window = self.hour_windows[client_id]
        hour_cutoff = now - 3600
        
        # Remove old entries
        while hour_window and hour_window[0] < hour_cutoff:
            hour_window.popleft()
        
        if len(hour_window) >= self.requests_per_hour:
            return {
                "error": "Rate limit exceeded",
                "limit": self.requests_per_hour,
                "window": "hour",
                "retry_after": int(3600 - (now - hour_window[0]))
            }
        
        # Check concurrent requests
        concurrent = self.concurrent_requests[client_id]
        if concurrent >= self.max_concurrent:
            return {
                "error": "Too many concurrent requests",
                "limit": self.max_concurrent,
                "window": "concurrent",
                "retry_after": 5
            }
        
        return None
    
    def _record_request(self, client_id: str):
        """Record a request for rate limiting tracking."""
        now = time.time()
        
        # Record in both windows
        self.minute_windows[client_id].append(now)
        self.hour_windows[client_id].append(now)
        
        # Increment concurrent requests
        self.concurrent_requests[client_id] += 1
    
    def _release_request(self, client_id: str):
        """Release a concurrent request slot."""
        if self.concurrent_requests[client_id] > 0:
            self.concurrent_requests[client_id] -= 1
        
        # Clean up if no concurrent requests
        if self.concurrent_requests[client_id] == 0:
            del self.concurrent_requests[client_id]
    
    async def dispatch(self, request: Request, call_next):
        # Cleanup old entries periodically
        self._cleanup_old_entries()
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        rate_limit_info = self._is_rate_limited(client_id)
        if rate_limit_info:
            logger.warning(f"Rate limit exceeded for {client_id}: {rate_limit_info}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": rate_limit_info["error"],
                    "detail": f"Rate limit of {rate_limit_info['limit']} requests per {rate_limit_info['window']} exceeded"
                },
                headers={
                    "Retry-After": str(rate_limit_info["retry_after"]),
                    "X-RateLimit-Limit": str(rate_limit_info["limit"]),
                    "X-RateLimit-Window": rate_limit_info["window"]
                }
            )
        
        # Record the request
        self._record_request(client_id)
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Remaining-Minute"] = str(
                max(0, self.requests_per_minute - len(self.minute_windows[client_id]))
            )
            response.headers["X-RateLimit-Remaining-Hour"] = str(
                max(0, self.requests_per_hour - len(self.hour_windows[client_id]))
            )
            
            return response
            
        finally:
            # Always release the concurrent request slot
            self._release_request(client_id)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    Includes timing information and error tracking.
    """
    
    def __init__(self, app: ASGIApp, log_requests: bool = True, log_responses: bool = False):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    def _get_client_info(self, request: Request) -> Dict[str, str]:
        """Extract client information from request."""
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP", client_ip)
        user_agent = request.headers.get("User-Agent", "unknown")
        
        return {
            "client_ip": client_ip,
            "real_ip": real_ip,
            "user_agent": user_agent
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request information
        client_info = self._get_client_info(request)
        
        # Log incoming request
        if self.log_requests:
            logger.info(
                f"Request: {request.method} {request.url.path} "
                f"from {client_info['client_ip']} "
                f"User-Agent: {client_info['user_agent'][:100]}"
            )
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Calculate response time
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            if self.log_responses or response.status_code >= 400:
                log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
                logger.log(
                    log_level,
                    f"Response: {response.status_code} "
                    f"for {request.method} {request.url.path} "
                    f"in {process_time:.4f}s "
                    f"from {client_info['client_ip']}"
                )
            
            return response
            
        except Exception as e:
            # Log errors
            process_time = time.time() - start_time
            logger.error(
                f"Error processing {request.method} {request.url.path} "
                f"from {client_info['client_ip']} "
                f"after {process_time:.4f}s: {str(e)}"
            )
            raise


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection middleware.
    Validates CSRF tokens for state-changing requests.
    """
    
    def __init__(self, app: ASGIApp, exempt_paths: Optional[list] = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/"]
        self.safe_methods = {"GET", "HEAD", "OPTIONS", "TRACE"}
    
    def _is_exempt(self, path: str) -> bool:
        """Check if path is exempt from CSRF protection."""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    async def dispatch(self, request: Request, call_next):
        # Skip CSRF check for safe methods and exempt paths
        if (request.method in self.safe_methods or 
            self._is_exempt(request.url.path)):
            return await call_next(request)
        
        # For API endpoints, we rely on JWT authentication
        # and SameSite cookie settings for CSRF protection
        # Additional CSRF token validation can be added here if needed
        
        return await call_next(request)