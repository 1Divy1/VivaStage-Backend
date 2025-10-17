from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from openai import OpenAI
from groq import Groq
from pathlib import Path

from app.core.logging import setup_logging
from app.controllers import shorts_generator_controller
from app.core.config import (
    OPENAI_API_KEY, 
    GROQ_API_KEY,
)

# Configure centralized logging
logger = setup_logging(
    log_level="INFO",
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
    title="Viva Stage",
    description="Video processing API for creating short-form content from YouTube videos",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

# Register API controllers
app.include_router(reel_jobs.router)

app.add_middleware(
      CORSMiddleware,
      allow_origins=["http://localhost:5173"],
      allow_credentials=True,
      allow_methods=["GET", "POST", "PUT", "DELETE"],
      allow_headers=["*"],
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
