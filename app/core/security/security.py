from jose import jwt, JWTError, ExpiredSignatureError
from jose.exceptions import JWTClaimsError
from fastapi import HTTPException, status

from app.core.security.jwks import get_jwks
from app.core.config import SUPABASE_AUDIENCE


def verify_jwt(token: str):
    try:
        jwks = get_jwks()
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