from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from openai import OpenAI
from groq import Groq
from pathlib import Path

from app.core.logging import setup_logging

from app.routes import reel_jobs
from app.config import (
    OPENAI_API_KEY, 
    GROQ_API_KEY, 
    CORS_ORIGINS, 
    ENABLE_SECURITY_HEADERS, 
    ENABLE_RATE_LIMITING,
    ENABLE_REQUEST_LOGGING,
    RATE_LIMIT_REQUESTS_PER_MINUTE,
    RATE_LIMIT_REQUESTS_PER_HOUR,
    DEBUG
)
from app.middleware.security_middleware import (
    SecurityHeadersMiddleware,
    RateLimitMiddleware, 
    RequestLoggingMiddleware,
    CSRFProtectionMiddleware
)

# Configure centralized logging
logger = setup_logging(
    log_level="DEBUG" if DEBUG else "INFO",
    log_file=Path("app.log")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown tasks."""
    logger.info("Starting Viva Stage AI Backend...")
    
    try:
        # Initialize API clients (singletons)
        app.state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        app.state.groq_client = Groq(api_key=GROQ_API_KEY)
        
        logger.info("API clients initialized successfully")
        
        # Startup validation
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not configured")
        if not GROQ_API_KEY:
            logger.warning("Groq API key not configured")
            
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
        
    finally:
        # Shutdown tasks
        logger.info("Application shutting down...")


# Create FastAPI instance with comprehensive configuration
app = FastAPI(
    title="Viva Stage AI",
    description="Video processing API for creating short-form content from YouTube videos",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    openapi_url="/openapi.json" if DEBUG else None
)


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging and response format."""
    logger.warning(
        f"HTTP {exc.status_code} error for {request.method} {request.url.path}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        },
        headers=getattr(exc, 'headers', None)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed error messages."""
    logger.warning(
        f"Validation error for {request.method} {request.url.path}: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Request validation failed",
            "details": exc.errors(),
            "status_code": 422,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with proper logging."""
    logger.error(
        f"Unexpected error for {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


# Add security middleware (order matters - add in reverse order of execution)
if ENABLE_REQUEST_LOGGING:
    app.add_middleware(RequestLoggingMiddleware, log_requests=True, log_responses=DEBUG)

if ENABLE_RATE_LIMITING:
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=RATE_LIMIT_REQUESTS_PER_MINUTE,
        requests_per_hour=RATE_LIMIT_REQUESTS_PER_HOUR
    )

# CSRF protection middleware
app.add_middleware(
    CSRFProtectionMiddleware,
    exempt_paths=["/docs", "/redoc", "/openapi.json", "/", "/health"]
)

if ENABLE_SECURITY_HEADERS:
    app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware (should be last middleware added)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-Remaining-Minute", "X-RateLimit-Remaining-Hour"]
)

# Mount static files for serving videos (with authentication middleware applied)
app.mount("/videos", StaticFiles(directory="output"), name="videos")

# Register API routes
app.include_router(reel_jobs.router)


@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {
        "message": "Welcome to Viva Stage AI!",
        "version": "1.0.0",
        "description": "Video processing API for creating short-form content",
        "docs": "/docs" if DEBUG else "Documentation disabled in production",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    return {
        "status": "healthy",
        "service": "viva-stage-ai-backend",
        "version": "1.0.0"
    }


@app.get("/auth/test")
async def auth_test():
    """Test endpoint to verify authentication is working (DEBUG only)."""
    if not DEBUG:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found"
        )
    
    return {
        "message": "Authentication test endpoint",
        "note": "This endpoint is only available in debug mode"
    }

