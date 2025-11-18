from jose import jwt, JWTError, ExpiredSignatureError
from jose.exceptions import JWTClaimsError
from fastapi import HTTPException, status

from app.core.security.jwks import get_jwks
from app.core.config import SUPABASE_AUDIENCE


async def verify_jwt(token: str):
    """
    Verify and decode a JWT token using Supabase JWKS.

    Args:
        token: JWT token string to verify

    Returns:
        dict: Decoded JWT payload

    Raises:
        HTTPException: If token is invalid, expired, or has invalid claims
    """
    try:
        jwks = await get_jwks()
        unverified_header = jwt.get_unverified_header(token)

        kid = unverified_header.get("kid")
        key = next((jwk for jwk in jwks["keys"] if jwk["kid"] == kid), None)

        if not key:
            raise HTTPException(status_code=401, detail="Invalid signing key")

        payload = jwt.decode(
            token,
            key,
            algorithms=[key["alg"]],
            audience=SUPABASE_AUDIENCE,
        )
        return payload
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTClaimsError:
        raise HTTPException(status_code=401, detail="Invalid claims")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")