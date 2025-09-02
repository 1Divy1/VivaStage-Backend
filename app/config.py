import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# load .env at startup
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# OpenAI and Groq API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Supabase configuration (New API Keys)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://rpymjflavcjhviaunmnc.supabase.co")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
SUPABASE_PROJECT_ID = SUPABASE_URL.split("//")[1].split(".")[0] if SUPABASE_URL else "rpymjflavcjhviaunmnc"

# JWT configuration for JWKS (using correct Supabase endpoint)
SUPABASE_JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "ES256")  # Updated for ECC P-256
JWT_AUDIENCE = "authenticated"  # Standard Supabase audience
JWT_ISSUER = f"{SUPABASE_URL}/auth/v1"

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

# Rate limiting configuration
RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
RATE_LIMIT_REQUESTS_PER_HOUR = int(os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", "1000"))

# Security settings
ENABLE_SECURITY_HEADERS = os.getenv("ENABLE_SECURITY_HEADERS", "true").lower() == "true"
ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true"
ENABLE_REQUEST_LOGGING = os.getenv("ENABLE_REQUEST_LOGGING", "true").lower() == "true"

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = ENVIRONMENT == "development"

# Validate required environment variables
def validate_config():
    """Validate that required environment variables are set."""
    required_vars = [
        ("SUPABASE_URL", SUPABASE_URL),
        ("GROQ_API_KEY", GROQ_API_KEY),
    ]
    
    missing_vars = [name for name, value in required_vars if not value]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )

# Auto-validate configuration on import
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        # Don't raise in production, just log the error
