import time
import requests
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError

class AuthService:
    def __init__(self, supabase_url: str, supabase_audience: str):
        self.supabase_url = supabase_url.rstrip("/")
        self.jwks_url = f"{self.supabase_url}/auth/v1/.well-known/jwks.json"
        self.audience = supabase_audience

        self.jwks = None
        self.jwks_expiry = 0  # epoch seconds
        self.cache_ttl = 600  # 10 minutes

    def _refresh_jwks(self):
        """Fetch JWKS from Supabase if cache expired."""
        if not self.jwks or time.time() > self.jwks_expiry:
            resp = requests.get(self.jwks_url, timeout=5)
            resp.raise_for_status()
            self.jwks = resp.json()
            self.jwks_expiry = time.time() + self.cache_ttl

    def validate_token(self, token: str) -> dict:
        """
        Validate a JWT from Supabase.
        Returns the decoded payload if valid.
        Raises JWTError on failure.
        """
        self._refresh_jwks()

        try:
            payload = jwt.decode(
                token,
                self.jwks,
                algorithms=["RS256", "ES256"],  # Supabase may use RS or ES
                audience=self.audience,
                issuer=f"{self.supabase_url}/auth/v1"
            )
            return payload

        except ExpiredSignatureError:
            raise JWTError("Token expired")
        except JWTClaimsError as e:
            raise JWTError(f"Invalid claims: {e}")
        except Exception as e:
            raise JWTError(f"Token validation failed: {e}")
