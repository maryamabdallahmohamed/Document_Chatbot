# config/settings.py
import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model configurations
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 50

# LLM configurations
OLLAMA_MODELS = {
    "default": "llama3:8b"
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"