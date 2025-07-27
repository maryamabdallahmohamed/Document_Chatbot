from sentence_transformers import CrossEncoder
import logging as logger
from tqdm import tqdm
from langchain.schema import Document

class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        self.reranker = CrossEncoder(model_name, cache_folder='./cache/reranker_cache')
        logger.info("✅ Reranker model loaded successfully")

    def rerank_chunks(self, query, chunks):
        """
        Rerank the retrieved chunks using BGE reranker.
        Returns chunks sorted by relevance score.
        """        
        logger.info(f"🔄 Reranking {len(chunks)} chunks...")
        
        # Prepare query-passage pairs for reranking
        pairs = [(query, chunk.page_content) for chunk in chunks]
        
        # Get relevance scores
        with tqdm(desc=f"🎯 Reranking {len(pairs)} query-chunk pairs", 
                bar_format="{desc} | {elapsed}"):
            scores = self.reranker.predict(pairs)
        
        # Combine chunks with scores and sort by relevance
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Update chunk metadata with reranking scores
        reranked_chunks = []
        for i, (chunk, score) in enumerate(scored_chunks):
            enhanced_metadata = chunk.metadata.copy() if chunk.metadata else {}
            enhanced_metadata["rerank_score"] = float(score)
            enhanced_metadata["rerank_position"] = i + 1
            
            enhanced_chunk = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            reranked_chunks.append(enhanced_chunk)
        
        logger.info(f"✅ Chunks reranked - Best score: {max(scores):.3f}")
        return reranked_chunks