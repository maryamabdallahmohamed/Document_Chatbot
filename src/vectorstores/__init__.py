"""Vector Store Implementations"""

try:
    from .fais_db import Fais_VS
except ImportError:
    try:
        from .faiss_vectorstore import Fais_VS
    except ImportError:
        pass

__all__ = []
if 'Fais_VS' in locals():
    __all__.append('Fais_VS')
