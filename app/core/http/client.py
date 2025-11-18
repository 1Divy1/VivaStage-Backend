"""
HTTP client abstraction layer for making HTTP requests.

This module provides a unified async HTTP client interface for all HTTP operations
in the application, built on top of httpx.
"""

from typing import Optional, Dict, Any
import httpx

from app.core.http.exceptions import (
    HTTPClientError,
    HTTPConnectionError,
    HTTPTimeoutError,
    HTTPStatusError
)


class HTTPClient:
    """
    Async HTTP client abstraction for making HTTP requests.

    This client provides a unified interface for all HTTP operations with:
    - Consistent error handling
    - Automatic exception wrapping
    - Configurable timeouts
    - Support for all common HTTP methods (GET, POST, PUT, DELETE, PATCH)

    Example:
        ```python
        client = HTTPClient()
        response = await client.get("https://api.example.com/data")
        data = response.json()
        ```
    """

    def __init__(self, default_timeout: float = 30.0):
        """
        Initialize HTTP client.

        Args:
            default_timeout: Default timeout in seconds for all requests (default: 30.0)
        """
        self.default_timeout = default_timeout

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> httpx.Response:
        """
        Make an async GET request.

        Args:
            url: The URL to request
            params: URL query parameters (optional)
            headers: HTTP headers to send (optional)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        return await self._request("GET", url, params=params, headers=headers, timeout=timeout)

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> httpx.Response:
        """
        Make an async POST request.

        Args:
            url: The URL to request
            json: JSON data to send in request body (optional)
            data: Form data or raw bytes to send (optional)
            headers: HTTP headers to send (optional)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        return await self._request("POST", url, json=json, data=data, headers=headers, timeout=timeout)

    async def put(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> httpx.Response:
        """
        Make an async PUT request.

        Args:
            url: The URL to request
            json: JSON data to send in request body (optional)
            data: Form data or raw bytes to send (optional)
            headers: HTTP headers to send (optional)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        return await self._request("PUT", url, json=json, data=data, headers=headers, timeout=timeout)

    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> httpx.Response:
        """
        Make an async DELETE request.

        Args:
            url: The URL to request
            headers: HTTP headers to send (optional)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        return await self._request("DELETE", url, headers=headers, timeout=timeout)

    async def patch(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> httpx.Response:
        """
        Make an async PATCH request.

        Args:
            url: The URL to request
            json: JSON data to send in request body (optional)
            data: Form data or raw bytes to send (optional)
            headers: HTTP headers to send (optional)
            timeout: Request timeout in seconds (uses default if not specified)

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        return await self._request("PATCH", url, json=json, data=data, headers=headers, timeout=timeout)

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """
        Internal method to make HTTP requests with unified error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            url: The URL to request
            **kwargs: Additional arguments to pass to httpx

        Returns:
            httpx.Response object

        Raises:
            HTTPConnectionError: If connection fails
            HTTPTimeoutError: If request times out
            HTTPStatusError: If server returns error status code
        """
        # Use default timeout if not specified
        if 'timeout' not in kwargs or kwargs['timeout'] is None:
            kwargs['timeout'] = self.default_timeout

        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

        except httpx.TimeoutException as e:
            raise HTTPTimeoutError(
                message=f"Request to {url} timed out after {kwargs['timeout']}s",
                url=url,
                original_error=e
            )

        except httpx.HTTPStatusError as e:
            raise HTTPStatusError(
                message=f"HTTP {e.response.status_code} error for {method} {url}",
                url=url,
                status_code=e.response.status_code,
                original_error=e
            )

        except httpx.RequestError as e:
            raise HTTPConnectionError(
                message=f"Connection failed for {method} {url}: {str(e)}",
                url=url,
                original_error=e
            )

        except Exception as e:
            raise HTTPClientError(
                message=f"Unexpected error during {method} {url}: {str(e)}",
                url=url,
                original_error=e
            )
