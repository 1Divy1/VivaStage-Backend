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
    GROQ_MODEL,
    LLM_PROVIDER,
    LOCAL_LLM_URL,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_TIMEOUT,
    TRANSCRIPTION_PROVIDER,
    GROQ_TRANSCRIPTION_MODEL
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
        app.state.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        app.state.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

        # Initialize shorts provider based on configuration
        from app.providers.llm.llm_provider_factory import LLMProviderFactory

        app.state.llm_provider = LLMProviderFactory.create_provider(
            provider_type=LLM_PROVIDER,
            openai_client=app.state.openai_client,
            groq_client=app.state.groq_client,
            groq_model=GROQ_MODEL,
            local_llm_url=LOCAL_LLM_URL,
            local_llm_model=LOCAL_LLM_MODEL,
            timeout=LOCAL_LLM_TIMEOUT
        )

        # Store transcription provider configuration for dependency injection
        app.state.transcription_provider_type = TRANSCRIPTION_PROVIDER
        app.state.groq_transcription_model = GROQ_TRANSCRIPTION_MODEL

        logger.info(f"LLM provider initialized: {LLM_PROVIDER}")
        logger.info(f"Transcription provider configured: {TRANSCRIPTION_PROVIDER}")
        logger.info("API clients initialized successfully")

        # Startup validation - fail fast with clear errors
        if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is not configured. "
                "Please set OPENAI_API_KEY in your environment variables or .env file."
            )
        if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is not configured. "
                "Please set GROQ_API_KEY in your environment variables or .env file."
            )
        if LLM_PROVIDER not in ["openai", "groq", "local"]:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. "
                "Supported values: 'openai', 'groq', 'local'"
            )

        # Transcription provider validation
        if TRANSCRIPTION_PROVIDER == "groq" and not GROQ_API_KEY:
            raise ValueError(
                "TRANSCRIPTION_PROVIDER is set to 'groq' but GROQ_API_KEY is not configured. "
                "Please set GROQ_API_KEY in your environment variables or .env file."
            )
        if TRANSCRIPTION_PROVIDER not in ["groq"]:
            raise ValueError(
                f"Unknown TRANSCRIPTION_PROVIDER: {TRANSCRIPTION_PROVIDER}. "
                "Supported values: 'groq'"
            )

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
app.include_router(shorts_generator_controller.router)

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
