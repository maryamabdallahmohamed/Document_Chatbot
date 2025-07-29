import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    def __init__(self):
        logger.info("🔧 Initializing BasePreprocessor...")
        logger.info("📄 Setting up text splitter configuration...")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP, 
            length_function=lambda x: len(x.split()),
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=False,
            add_start_index=True,
            strip_whitespace=True
        )
        
        logger.info(f"✅ Text splitter configured - Chunk size: {DEFAULT_CHUNK_SIZE}, Overlap: {DEFAULT_CHUNK_OVERLAP}")
        logger.info("✅ BasePreprocessor initialized successfully")

    @abstractmethod
    def load_and_preprocess_data(self, file_path):
        pass

    @abstractmethod
    def process_documents_from_files(self, file_paths):
        pass

    def clean_text(self, text):
        """Clean text by normalizing whitespace and line breaks."""
        logger.debug("🧹 Cleaning text...")
        
        try:
            original_length = len(str(text))
            
            # Clean excessive line breaks and whitespace
            cleaned = re.sub(r'\s+', ' ', re.sub(r'\n{3,}', '\n\n', str(text))).strip()
            
            cleaned_length = len(cleaned)
            logger.debug(f"✅ Text cleaned - Original: {original_length} chars, Cleaned: {cleaned_length} chars")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Error cleaning text: {str(e)}")
            raise

    def chunk_documents(self, individual_documents):
        """Split documents into chunks using the configured text splitter."""
        logger.info(f"✂️ Starting document chunking for {len(individual_documents)} documents...")
        
        try:
            chunked_docs = []
            total_chunks = 0
            
            # Process each document with progress tracking
            for doc in tqdm(individual_documents, desc="📄 Processing docs", unit="doc"):
                logger.debug(f"🔄 Chunking document with pdf_id: {doc.metadata.get('pdf_id', 'unknown')}")
                
                # Split document text into chunks
                chunks = self.text_splitter.split_text(doc.page_content)
                logger.debug(f"📊 Document split into {len(chunks)} chunks")
                
                # Create Document objects for each chunk with progress
                for i, chunk in enumerate(tqdm(chunks, desc=f"🔗 Creating chunks", unit="chunk", leave=False)):
                    chunked_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "pdf_id": doc.metadata["pdf_id"],
                                "chunk_id": i
                            }
                        )
                    )
                
                total_chunks += len(chunks)
                logger.debug(f"✅ Processed document - Created {len(chunks)} chunks (Total so far: {total_chunks})")

            logger.info(f"✅ Document chunking completed successfully")
            logger.info(f"📊 Total Documents: {len(individual_documents)}, Total Chunks: {len(chunked_docs)}")
            print(f"✅ Total Chunks: {len(chunked_docs)}")

            return chunked_docs
            
        except Exception as e:
            logger.error(f"❌ Error during document chunking: {str(e)}")
            raise