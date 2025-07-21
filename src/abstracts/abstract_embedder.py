from abc import ABC, abstractmethod
import torch
from langchain_huggingface import HuggingFaceEmbeddings

class Embedder(ABC): 
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.batch_size = batch_size
        
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={'device': self.device},encode_kwargs={'normalize_embeddings': True},
                                                     show_progress=True,cache_folder='./embedder_model_cache')

    @abstractmethod
    def embed_documents(self, documents):
        pass

    @abstractmethod
    def batch_embed(self, texts, batch_size=None): 
        pass
