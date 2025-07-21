from src.abstracts.abstract_embedder import Embedder

import numpy as np
import logging
from tqdm import tqdm


logger = logging.getLogger(__name__)

class MultilingualEmbedder(Embedder): 
    def __init__(self, model_name, batch_size):        
        super().__init__(model_name, batch_size)    
        logger.info("✅ MultilingualEmbedder initialized successfully")

    def embed_documents(self, documents):
        """Embed a list of documents using batch processing."""
        logger.info(f"📄 Embedding {len(documents)} documents...")
        logger.info(f"📦 Using batch size: {self.batch_size}")
        
        try:
            with tqdm(total=1, desc="📝 Processing documents", unit="batch") as pbar:
                embeddings = self.batch_embed(documents, batch_size=self.batch_size)
                pbar.update(1)
            
            logger.info(f"✅ Document embedding completed - Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Error embedding documents: {str(e)}")
            raise

    def batch_embed(self, texts, batch_size=None):
        """Process texts in batches to generate embeddings."""
        if batch_size is None:
            batch_size = self.batch_size
        
        logger.info(f"⚡ Starting batch embedding for {len(texts)} texts...")
        logger.info(f"📦 Batch size: {batch_size}")
        
        # Calculate number of batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"🔢 Processing {num_batches} batches...")
        
        try:
            embeddings = []
            
            # Process texts in batches with progress tracking
            for i in tqdm(range(0, len(texts), batch_size), desc="⚡ Embedding batches", unit="batch"):
                batch = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                logger.debug(f"🔄 Processing batch {batch_num}/{num_batches} - {len(batch)} texts")
                
                try:
                    # Generate embeddings for current batch
                    with tqdm(total=1, desc=f"🧠 Batch {batch_num}", unit="embedding", leave=False) as batch_pbar:
                        batch_embeddings = self.embedding_model.embed_documents(batch)
                        batch_pbar.update(1)
                    
                    embeddings.extend(batch_embeddings)
                    logger.debug(f"✅ Batch {batch_num} completed - {len(batch_embeddings)} embeddings generated")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch {batch_num}: {str(e)}")
                    raise
            
            # Convert to numpy array
            logger.info("🔄 Converting embeddings to numpy array...")
            result = np.array(embeddings, dtype=np.float32)
            
            logger.info(f"✅ Batch embedding completed successfully!")
            logger.info(f"📊 Final embedding shape: {result.shape}")
            logger.info(f"📈 Total embeddings generated: {len(embeddings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {str(e)}")
            if "memory" in str(e).lower():
                logger.error(f"💡 Suggestion: Try reducing batch size from {batch_size} to {batch_size // 2}")
            elif "model" in str(e).lower():
                logger.error("💡 Suggestion: Check if embedding model is properly loaded")
            raise