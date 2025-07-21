from abc import ABC, abstractmethod
import torch


class BaseLLM(ABC):
    def __init__(self, model_name, cache_folder):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )

    @abstractmethod
    def load_model(self):
        pass
