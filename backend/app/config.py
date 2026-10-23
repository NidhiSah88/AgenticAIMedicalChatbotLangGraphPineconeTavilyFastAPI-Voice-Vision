
# import os 
# import certifi 
# from dotenv import load_dotenv

# # --- AUTOMATIC SSL FIX FOR CONDA/PYTHON ENVIRONMENTS ---
# ssl_cert = os.environ.get('SSL_CERT_FILE')
# if ssl_cert and not os.path.exists(ssl_cert):
#     os.environ['SSL_CERT_FILE'] = certifi.where()
# elif not ssl_cert:
#     os.environ['SSL_CERT_FILE'] = certifi.where()
# # --------------------------------------------------------

# load_dotenv()
# from pydantic_settings import BaseSettings, SettingsConfigDict

# class Settings(BaseSettings):
#     openai_api_key: str = ''
#     pinecone_api_key: str = ''
#     tavily_api_key: str = ''
#     pinecone_index: str = "medical-agentic-chatbot"
#     openai_chat_model: str = 'gpt-4.1-mini'
#     openai_embed_model: str = 'text-embedding-3-small'
#     openai_vision_model: str = 'gpt-4.1-mini'
#     api_token: str = ''
#     allowed_origins: str = 'http://localhost:8000,http://127.0.0.1:8000'
#     model_config = SettingsConfigDict(env_file='.env', extra='ignore')



# settings = Settings()


# Your current settings class uses OpenAI-specific configuration. Replace it with the following.


import os
import certifi
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load the .env file located in the backend directory.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Configure SSL certificates for API connections.
ssl_cert = os.environ.get("SSL_CERT_FILE")

if not ssl_cert or not os.path.exists(ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()


class Settings(BaseSettings):
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

    api_token: str = ""
    allowed_origins: str = (
        "http://localhost:3000,http://localhost:8000"
    )

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()



