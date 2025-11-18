"""
HTTP client abstraction layer.

This module provides a unified async HTTP client and exception handling
for all HTTP operations in the application.
"""

from app.core.http.client import HTTPClient
from app.core.http.exceptions import (
    HTTPClientError,
    HTTPConnectionError,
    HTTPTimeoutError,
    HTTPStatusError
)

__all__ = [
    "HTTPClient",
    "HTTPClientError",
    "HTTPConnectionError",
    "HTTPTimeoutError",
    "HTTPStatusError",
]
