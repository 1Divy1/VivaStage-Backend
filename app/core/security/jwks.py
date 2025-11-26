import time
from app.core.config import SUPABASE_JWKS_URL
from app.core.http import HTTPClient

# TODO: TO BE IMPLEMENTED

# simple in-memory cache
_jwks_cache = None
_jwks_timestamp = 0
CACHE_TTL = 600  # 10 minutes

# HTTP client instance
_http_client = HTTPClient(default_timeout=10.0)

async def get_jwks():
    """
    Fetch JSON Web Key Set from Supabase for JWT validation.

    Implements a simple in-memory cache with 10-minute TTL to reduce API calls.

    Returns:
        dict: JWKS data containing public keys for JWT verification

    Raises:
        HTTPClientError: If the request to fetch JWKS fails
    """
    global _jwks_cache, _jwks_timestamp
    now = time.time()

    if _jwks_cache is None or now - _jwks_timestamp > CACHE_TTL:
        resp = await _http_client.get(SUPABASE_JWKS_URL)
        _jwks_cache = resp.json()
        _jwks_timestamp = now

    return _jwks_cache
