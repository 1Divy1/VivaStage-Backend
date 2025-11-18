"""
Authentication dependencies for FastAPI endpoints.
Provides JWT validation and user context extraction.
"""
from fastapi import HTTPException, Header
from app.core.security.security import verify_jwt


async def get_current_user(authorization: str = Header(...)):
    """
    FastAPI dependency to extract and verify current user from JWT token.

    Args:
        authorization: Authorization header with Bearer token

    Returns:
        dict: Verified JWT payload containing user information

    Raises:
        HTTPException: If authorization header is invalid or token verification fails
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid header")

    token = authorization.split(" ")[1]
    return await verify_jwt(token)
