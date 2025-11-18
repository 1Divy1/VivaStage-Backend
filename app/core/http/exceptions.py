"""
Custom exceptions for HTTP client operations.

This module provides a hierarchy of exceptions for handling HTTP-related errors
in a consistent way across the application.
"""

from typing import Optional


class HTTPClientError(Exception):
    """
    Base exception for all HTTP client errors.

    This is the parent class for all HTTP-related exceptions in the application.
    Catch this to handle any HTTP error generically.
    """

    def __init__(
        self,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize HTTP client error.

        Args:
            message: Human-readable error description
            url: The URL that was being accessed (optional)
            status_code: HTTP status code if applicable (optional)
            original_error: The underlying exception that caused this error (optional)
        """
        self.message = message
        self.url = url
        self.status_code = status_code
        self.original_error = original_error

        super().__init__(self.message)

    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [self.message]
        if self.url:
            parts.append(f"URL: {self.url}")
        if self.status_code:
            parts.append(f"Status: {self.status_code}")
        return " | ".join(parts)


class HTTPConnectionError(HTTPClientError):
    """
    Exception raised when connection to the server fails.

    This includes DNS resolution failures, connection timeouts,
    connection refused errors, etc.
    """
    pass


class HTTPTimeoutError(HTTPClientError):
    """
    Exception raised when a request times out.

    This occurs when the server doesn't respond within the specified timeout period.
    """
    pass


class HTTPStatusError(HTTPClientError):
    """
    Exception raised when the server returns an error status code.

    This includes 4xx client errors and 5xx server errors.
    """
    pass
