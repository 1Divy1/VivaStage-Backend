"""
Authentication dependencies for FastAPI endpoints.
Provides JWT validation and user context extraction.
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

# HTTP Bearer token security scheme
security = HTTPBearer(
    scheme_name="JWT",
    description="JWT token from Supabase authentication"
)


class CurrentUser:
    """Class to hold current user context."""
    
    def __init__(self, user_data: Dict[str, Any]):
        self.user_id: str = user_data.get('user_id')
        self.email: str = user_data.get('email')
        self.role: str = user_data.get('role', 'authenticated')
        self.app_metadata: Dict[str, Any] = user_data.get('app_metadata', {})
        self.user_metadata: Dict[str, Any] = user_data.get('user_metadata', {})
        self.aud: str = user_data.get('aud')
        self.exp: int = user_data.get('exp')
        self.iss: str = user_data.get('iss')
    
    def __str__(self) -> str:
        return f"User(id={self.user_id}, email={self.email}, role={self.role})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == 'admin' or self.app_metadata.get('role') == 'admin'
    
    def is_premium(self) -> bool:
        """Check if user has premium subscription."""
        return (
            self.app_metadata.get('subscription_status') == 'active' or
            self.user_metadata.get('is_premium', False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user context to dictionary."""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'role': self.role,
            'app_metadata': self.app_metadata,
            'user_metadata': self.user_metadata,
            'aud': self.aud,
            'exp': self.exp,
            'iss': self.iss
        }


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> CurrentUser:
    """
    Dependency to get current authenticated user.
    Validates JWT token and returns user context.
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        CurrentUser: Current user context
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not credentials:
        logger.warning("No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    if not token:
        logger.warning("Empty token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Extract user context from token
        user_data = await auth_service.extract_user_context(token)
        user = CurrentUser(user_data)
        
        logger.info(f"User authenticated successfully: {user.user_id}")
        return user
        
    except HTTPException:
        # Re-raise HTTP exceptions from auth service
        raise
    except Exception as e:
        logger.error(f"Unexpected error in authentication: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[CurrentUser]:
    """
    Optional dependency to get current authenticated user.
    Returns None if no valid token is provided.
    
    Args:
        credentials: Optional HTTP Bearer token credentials
        
    Returns:
        Optional[CurrentUser]: Current user context or None
    """
    if not credentials or not credentials.credentials:
        return None
    
    try:
        user_data = await auth_service.extract_user_context(credentials.credentials)
        return CurrentUser(user_data)
    except HTTPException:
        # Return None for invalid tokens in optional auth
        return None
    except Exception:
        # Return None for any errors in optional auth
        return None


async def get_admin_user(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Dependency to ensure current user is an admin.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        CurrentUser: Current admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin():
        logger.warning(f"User {current_user.user_id} attempted admin action without permission")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


async def get_premium_user(
    current_user: CurrentUser = Depends(get_current_user)
) -> CurrentUser:
    """
    Dependency to ensure current user has premium access.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        CurrentUser: Current premium user
        
    Raises:
        HTTPException: If user doesn't have premium access
    """
    if not current_user.is_premium():
        logger.warning(f"User {current_user.user_id} attempted premium action without subscription")
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Premium subscription required"
        )
    
    return current_user


def require_user_id(user_id: str):
    """
    Dependency factory to ensure current user matches a specific user ID.
    Useful for protecting user-specific resources.
    
    Args:
        user_id: Required user ID
        
    Returns:
        Dependency function
    """
    async def _require_user_id(
        current_user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        if current_user.user_id != user_id and not current_user.is_admin():
            logger.warning(
                f"User {current_user.user_id} attempted to access resource for user {user_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient permissions"
            )
        return current_user
    
    return _require_user_id


class RateLimitInfo:
    """Class to hold rate limit information."""
    
    def __init__(self, user_id: str, is_premium: bool = False):
        self.user_id = user_id
        self.is_premium = is_premium
        
        # Set rate limits based on user type
        if is_premium:
            self.requests_per_minute = 100
            self.requests_per_hour = 1000
            self.concurrent_requests = 10
        else:
            self.requests_per_minute = 20
            self.requests_per_hour = 200
            self.concurrent_requests = 3


async def get_rate_limit_info(
    current_user: CurrentUser = Depends(get_current_user)
) -> RateLimitInfo:
    """
    Dependency to get rate limit information for current user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        RateLimitInfo: Rate limit configuration for user
    """
    return RateLimitInfo(
        user_id=current_user.user_id,
        is_premium=current_user.is_premium()
    )