import logging
import time
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Setup path and imports
SRC_ROOT = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# External library imports
from langchain_core.documents import Document

# Internal imports
from src.models.multilingual_embedder import MultilingualEmbedder
from src.models.llm_models import OLLAMA_LLM
from src.vectorstores.faiss_vectorstore import Fais_VS
from src.strategies.chat_strategy import ChattingStrategy
from src.strategies.question_strategy import QuestionStrategy
from src.strategies.summarization_strategy import SummarizationStrategy, Summarization_Rag_Strategy
from src.core.task_processor import TaskProcessor
from src.processors.json_processor import JSONPreprocessor


def setup_logging():
    """Configure logging for the pipeline."""
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/document_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_and_process_documents(logger,files_paths):
    """Load and process  documents into chunks."""
    logger.info("📁 Loading documents...")
    paths = [files_paths]
    docs = JSONPreprocessor()

    with tqdm(total=len(paths), desc="🔄 Processing  files", unit="file") as pbar:
        data = docs.process_documents_from_files(paths)
        pbar.update(len(paths))

    logger.info(f"✅ Loaded {len(data)} documents")

    logger.info("📄 Creating individual documents...")
    individual_documents = [
        Document(page_content=pdf.page_content, metadata={"pdf_id": i})
        for i, pdf in enumerate(tqdm(data, desc="🔨 Creating documents", unit="doc"))
        if pdf.page_content
    ]
    logger.info(f"✅ Created {len(individual_documents)} individual documents")

    logger.info("✂️ Chunking documents...")
    chunked_docs = docs.chunk_documents(individual_documents)
    logger.info(f"✅ Document chunking completed - {len(chunked_docs)} chunks created")

    return chunked_docs , individual_documents


def initialize_models(logger, pipeline_start_time):
    """Initialize embedder and LLM models."""
    logger.info("🧠 Initializing multilingual embedder model...")
    with tqdm(total=1, desc="⚡ Loading embedder", unit="model") as pbar:
        multilingual_embedder = MultilingualEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            batch_size=32
        )
        pbar.update(1)

    current_elapsed = time.time() - pipeline_start_time
    logger.info(f"✅ Embedder model loaded - Total elapsed: {current_elapsed:.2f}s")
    print(f"⏱️ Time Taken to process embedder: {current_elapsed:.2f}s")

    logger.info("🤖 Loading OLLAMA LLM model...")
    with tqdm(total=1, desc="🔥 Loading LLM", unit="model") as pbar:
        llm = OLLAMA_LLM('llama3:8b', './cache/llm_cache').load_model()
        pbar.update(1)

    current_elapsed = time.time() - pipeline_start_time
    logger.info(f"✅ LLM model loaded - Total elapsed: {current_elapsed:.2f}s")
    print(f"⏱️ Time Taken to process LLM: {current_elapsed:.2f}s")

    return multilingual_embedder, llm


def create_vector_store(logger, chunked_docs, multilingual_embedder):
    """Create and populate vector store with document embeddings."""
    logger.info("🗄️ Creating vector store...")
    vector_store = Fais_VS()
    vector_store.set_embedder_model(multilingual_embedder)

    with tqdm(total=len(chunked_docs), desc="🔢 Creating embeddings", unit="chunk") as pbar:
        vector_store.create_vector_store(chunked_docs)
        pbar.update(len(chunked_docs))

    logger.info(f"✅ Vector store created with {len(chunked_docs)} embeddings")
    return vector_store


def initialize_strategies(logger, llm, vector_store, multilingual_embedder, pipeline_start_time):
    """Initialize all processing strategies."""
    logger.info("⚙️ Initializing processing strategies...")
    strategies_progress = tqdm(total=5, desc="🔧 Initializing strategies", unit="strategy")

    logger.info("💬 Initializing Chatting Strategy...")
    chatting_strategy = ChattingStrategy(llm, vector_store, multilingual_embedder)
    strategies_progress.update(1)

    logger.info("📝 Initializing Summarization Strategy...")
    summarization_strategy = SummarizationStrategy(llm)
    strategies_progress.update(1)

    logger.info("❓ Initializing Question Strategy...")
    question_strategy = QuestionStrategy(llm)
    strategies_progress.update(1)

    logger.info("🔍 Initializing RAG Summary Strategy...")
    rag_summary = Summarization_Rag_Strategy(llm, vector_store)
    strategies_progress.update(1)

    logger.info("🎯 Initializing Task Processor...")
    processor = TaskProcessor()
    strategies_progress.update(1)
    strategies_progress.close()

    current_elapsed = time.time() - pipeline_start_time
    logger.info(f"✅ All strategies initialized - Total elapsed: {current_elapsed:.2f}s")
    print(f"⏱️ Time Taken to process strategies: {current_elapsed:.2f}s")

    return chatting_strategy, summarization_strategy , question_strategy , rag_summary , processor




def main():
    """Main pipeline execution function."""
    pipeline_start_time = time.time()
    logger = setup_logging()
    logger.info("🚀 Starting document processing pipeline...")

    try:
        # Load and process documents
        chunked_docs , individual_documents = load_and_process_documents(logger,'data/Market Research Report_extracted_text.json')

        # Initialize models
        multilingual_embedder, llm = initialize_models(logger, pipeline_start_time)

        # Create vector store
        vector_store = create_vector_store(logger, chunked_docs, multilingual_embedder)

        # Initialize strategies
        chatting_strategy, summarization_strategy , question_strategy , rag_summary , processor = initialize_strategies(
            logger, llm, vector_store, multilingual_embedder, pipeline_start_time
        )
        processor.strategy =chatting_strategy
        processor.execute_task("\nWhat tranlation platforms are in the document\n")
        processor.strategy =summarization_strategy
        processor.execute_task(individual_documents[0].page_content,length="medium", verbose=False, overview_level='low_level')
        processor.strategy=question_strategy
        processor.execute_task(individual_documents[0],20,'hard')
        processor.strategy=rag_summary
        processor.execute_task("Translation Platforms")

 

        # Final timing
        final_elapsed_time = time.time() - pipeline_start_time
        logger.info(f"✅ Pipeline completed - Total time: {final_elapsed_time:.2f}s")
        print(f"🎉 Pipeline completed successfully in {final_elapsed_time:.2f}s!")

    

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        raise


if __name__ == "__main__":
    result = main()