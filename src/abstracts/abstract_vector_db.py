from abc import ABC, abstractmethod

class VectorStoreBase(ABC):
    @abstractmethod
    def create_vector_store(self, documents, embedder_model):
        pass
    
    @abstractmethod
    def get_relevant_documents(self, query, top_k=5):
        pass
    
    @abstractmethod
    def save_index(self, file_path):
        pass
    
    @abstractmethod
    def load_index(self, file_path):
        pass

