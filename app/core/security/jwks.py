import requests
import time
from app.core.config import SUPABASE_JWKS_URL

# simple in-memory cache
_jwks_cache = None
_jwks_timestamp = 0
CACHE_TTL = 600  # 10 minutes

def get_jwks():
    global _jwks_cache, _jwks_timestamp
    now = time.time()

    if _jwks_cache is None or now - _jwks_timestamp > CACHE_TTL:
        resp = requests.get(SUPABASE_JWKS_URL)
        resp.raise_for_status()
        _jwks_cache = resp.json()
        _jwks_timestamp = now

    return _jwks_cache
