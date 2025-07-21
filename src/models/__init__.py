"""Models Package"""

try:
    from .llm import OLLAMA_LLM, Hugging_Face_LLM
except ImportError:
    try:
        from .llm_models import OLLAMA_LLM, Hugging_Face_LLM
    except ImportError:
        pass

try:
    from .minilm_embedder import MultilingualEmbedder
except ImportError:
    try:
        from .multilingual_embedder import MultilingualEmbedder
    except ImportError:
        pass

__all__ = []
if 'OLLAMA_LLM' in locals():
    __all__.extend(['OLLAMA_LLM', 'Hugging_Face_LLM'])
if 'MultilingualEmbedder' in locals():
    __all__.append('MultilingualEmbedder')
