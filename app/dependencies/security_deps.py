"""
Authentication dependencies for FastAPI endpoints.
Provides JWT validation and user context extraction.
"""
from fastapi import HTTPException, Header
from app.core.security.security import verify_jwt


def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid header")

    token = authorization.split(" ")[1]
    return verify_jwt(token)
