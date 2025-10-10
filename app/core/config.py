import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

# load .env at startup
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# 3rd party API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL")
SUPABASE_AUDIENCE = os.getenv("SUPABASE_AUDIENCE")
