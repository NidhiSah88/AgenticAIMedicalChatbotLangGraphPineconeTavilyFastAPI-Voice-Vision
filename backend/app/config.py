

# Loads API keys, model names, and application settings	
# Run : During backend imports/startup


import os
import certifi
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Find the backend directory.
# __file__ is the path to config.py.
# parent = app directory; parent.parent = backend directory.
# Load the .env file located in the backend directory.
BASE_DIR = Path(__file__).resolve().parent.parent


# Load backend/.env into environment variables.
load_dotenv(BASE_DIR / ".env")

# Configure SSL certificates for API connections.
ssl_cert = os.environ.get("SSL_CERT_FILE")


# If the certificate is missing or invalid, use certifi's
# trusted CA certificate bundle for HTTPS connections.
if not ssl_cert or not os.path.exists(ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()


class Settings(BaseSettings):

    # Groq API key for chat, vision, transcription, and speech.

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_transcription_model: str = "whisper-large-v3-turbo"
    groq_vision_model: str = "qwen/qwen3.8-27b"


    pinecone_api_key: str = ""
    pinecone_index: str = "medical-agentic-chatbot-local384"

    tavily_api_key: str = ""

    embed_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
     # Optional bearer token used to protect API endpoints.
    api_token: str = ""

    # Comma-separated frontend/backend origins allowed by CORS.
    allowed_origins: str = (
        "http://localhost:3000,http://localhost:8000"
    )

    # Tell Pydantic where to load configuration from.
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

# 1. settings = Settings() creates the object used by the rest of the application.
settings = Settings()



