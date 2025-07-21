from abc import ABC, abstractmethod
class TaskStrategy(ABC):
    """Abstract base class defining the strategy interface."""
    
    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute the strategy. Must be implemented by concrete strategies."""
        pass