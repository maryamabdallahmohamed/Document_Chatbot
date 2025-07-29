# config/settings.py
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configurations
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# LLM configurations
OLLAMA_MODELS = {
    "qwen3": os.getenv("QWEN3_MODEL", "qwen3:8b")
}

# API Server configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "5050"))
API_DEBUG = os.getenv("API_DEBUG", "True").lower() == "true"

# File upload settings
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "data")
ALLOWED_EXTENSIONS = {'json', 'txt', 'pdf'}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_SIZE", "16777216"))  # 16MB default

# Retrieval settings
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
FUZZY_THRESHOLD = int(os.getenv("FUZZY_THRESHOLD", "70"))

# Cache settings
EMBEDDER_CACHE_DIR = CACHE_DIR / "embedder_model_cache"
RERANKER_CACHE_DIR = CACHE_DIR / "reranker_cache"
LLM_CACHE_DIR = CACHE_DIR / "llm_cache"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Reranker configuration
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Conversation settings
DEFAULT_CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "6"))