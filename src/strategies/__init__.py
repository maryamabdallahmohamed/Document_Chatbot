"""Strategy Implementations"""

try:
    from .chatting_module import ChattingStrategy
except ImportError:
    try:
        from .chat_strategy import ChattingStrategy
    except ImportError:
        pass

try:
    from .Question_module import QuestionStrategy
except ImportError:
    try:
        from .question_strategy import QuestionStrategy
    except ImportError:
        pass

try:
    from .Summerization_module import SummarizationStrategy, Summarization_Rag_Strategy
except ImportError:
    try:
        from .summarization_strategy import SummarizationStrategy, Summarization_Rag_Strategy
    except ImportError:
        pass

__all__ = []
if 'ChattingStrategy' in locals():
    __all__.append('ChattingStrategy')
if 'QuestionStrategy' in locals():
    __all__.append('QuestionStrategy')
if 'SummarizationStrategy' in locals():
    __all__.extend(['SummarizationStrategy', 'Summarization_Rag_Strategy'])
