from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback

# Setup path and imports (same as your bot.py)
SRC_ROOT = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_ROOT))

# External library imports
from langchain_core.documents import Document

# Internal imports (matching your bot.py structure)
from src.models.multilingual_embedder import MultilingualEmbedder
from src.models.llm_models import OLLAMA_LLM
from src.vectorstores.faiss_vectorstore import Fais_VS
from src.strategies.chat_strategy import ChattingStrategy
from src.strategies.question_strategy import QuestionStrategy
from src.strategies.summarization_strategy import SummarizationStrategy, Summarization_Rag_Strategy
from src.core.task_processor import TaskProcessor
from src.processors.json_processor import JSONPreprocessor

app = Flask(__name__, 
           template_folder='frontend',
           static_folder='frontend')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'json', 'txt', 'pdf'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentChatbotAPI:
    def __init__(self):
        self.processor = None
        self.chatting_strategy = None
        self.summarization_strategy = None
        self.question_strategy = None
        self.rag_summary = None
        self.individual_documents = None
        self.chunked_docs = None
        self.vector_store = None
        self.llm = None
        self.multilingual_embedder = None
        self.current_document_path = None
        self.is_initialized = False
        
    def setup_logging(self):
        """Configure logging for the pipeline."""
        return logging.getLogger(__name__)

    def load_and_process_documents(self, files_paths):
        """Load and process documents into chunks (adapted from your bot.py)."""
        logger.info("📁 Loading documents...")
        paths = [files_paths]
        docs = JSONPreprocessor()

        data = docs.process_documents_from_files(paths)
        logger.info(f"✅ Loaded {len(data)} documents")

        logger.info("📄 Creating individual documents...")
        individual_documents = [
            Document(page_content=pdf.page_content, metadata={"pdf_id": i})
            for i, pdf in enumerate(data)
            if pdf.page_content
        ]
        logger.info(f"✅ Created {len(individual_documents)} individual documents")

        logger.info("✂️ Chunking documents...")
        chunked_docs = docs.chunk_documents(individual_documents)
        logger.info(f"✅ Document chunking completed - {len(chunked_docs)} chunks created")

        return chunked_docs, individual_documents

    def initialize_models(self):
        """Initialize embedder and LLM models."""
        logger.info("🧠 Initializing multilingual embedder model...")
        self.multilingual_embedder = MultilingualEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            batch_size=32
        )
        logger.info("✅ Embedder model loaded")

        logger.info("🤖 Loading OLLAMA LLM model...")
        self.llm = OLLAMA_LLM('qwen3:8b', './cache/qwen3:8b_cache').load_model()
        logger.info("✅ LLM model loaded")

    def create_vector_store(self):
        """Create and populate vector store with document embeddings."""
        logger.info("🗄️ Creating vector store...")
        self.vector_store = Fais_VS()
        self.vector_store.set_embedder_model(self.multilingual_embedder)
        self.vector_store.create_vector_store(self.chunked_docs)
        logger.info(f"✅ Vector store created with {len(self.chunked_docs)} embeddings")

    def initialize_strategies(self):
        """Initialize all processing strategies."""
        logger.info("⚙️ Initializing processing strategies...")
        
        self.chatting_strategy = ChattingStrategy(self.llm, self.vector_store, self.multilingual_embedder)
        self.summarization_strategy = SummarizationStrategy(self.llm)
        self.question_strategy = QuestionStrategy(self.llm)
        self.rag_summary = Summarization_Rag_Strategy(self.llm, self.vector_store)
        self.processor = TaskProcessor()
        
        logger.info("✅ All strategies initialized")

    def initialize_system(self, document_path):
        """Initialize the entire system with a document."""
        try:
            logger.info("🚀 Initializing document processing system...")
            
            # Load and process documents
            self.chunked_docs, self.individual_documents = self.load_and_process_documents(document_path)
            
            # Initialize models
            self.initialize_models()
            
            # Create vector store
            self.create_vector_store()
            
            # Initialize strategies
            self.initialize_strategies()
            
            self.current_document_path = document_path
            self.is_initialized = True
            logger.info("✅ System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            return False

    def execute_strategy(self, strategy_name, message, options=None):
        """Execute a strategy with given parameters."""
        if not self.is_initialized:
            return {"error": "System not initialized. Please upload a document first."}
        
        try:
            if options is None:
                options = {}
                
            if strategy_name == 'chatting_strategy':
                self.processor.strategy = self.chatting_strategy
                result = self.processor.execute_task(message)
                result=result['answer']
                
            elif strategy_name == 'rag_summary':
                self.processor.strategy = self.rag_summary
                # Use keywords from options or message as query
                query = options.get('keywords', message)
                result = self.processor.execute_task(query)
                
            elif strategy_name == 'question_strategy':
                if not self.individual_documents:
                    return {"error": "No document available for question generation"}
                
                count = int(options.get('count', 20))
                q_type = options.get('type', 'hard') if options.get('type') else 'hard'
                
                self.processor.strategy = self.question_strategy
                # Use first document
                doc_to_use = self.individual_documents[0]
                result = self.processor.execute_task(doc_to_use, count, q_type)
                result=result['qa_output']
                
            elif strategy_name == 'summarization':
                if not self.individual_documents:
                    return {"error": "No document available for summarization"}
                
                length = options.get('length', 'medium')
                verbose = options.get('verbose', False)
                overview_level = options.get('overview_level', 'low_level')
                
                self.processor.strategy = self.summarization_strategy
                # Use first document's content
                doc_content = self.individual_documents[0].page_content
                result = self.processor.execute_task(
                    doc_content, 
                    length=length, 
                    verbose=verbose, 
                    overview_level=overview_level
                )
                
            else:
                return {"error": f"Unknown strategy: {strategy_name}"}
            
            # Return raw result without any HTML formatting
            return {"response": str(result)}
            
        except Exception as e:
            logger.error(f"Strategy execution error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Strategy execution failed: {str(e)}"}

# Global API instance
chatbot_api = DocumentChatbotAPI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status."""
    return jsonify({
        'initialized': chatbot_api.is_initialized,
        'document_loaded': chatbot_api.current_document_path is not None,
        'current_document': chatbot_api.current_document_path,
        'total_documents': len(chatbot_api.individual_documents) if chatbot_api.individual_documents else 0,
        'total_chunks': len(chatbot_api.chunked_docs) if chatbot_api.chunked_docs else 0
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and system initialization."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload JSON, TXT, or PDF files.'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filepath}")
        
        # Initialize the system with the new document
        success = chatbot_api.initialize_system(filepath)
        
        if success:
            return jsonify({
                'message': f'File {filename} uploaded and processed successfully',
                'filename': filename,
                'filepath': filepath,
                'total_documents': len(chatbot_api.individual_documents),
                'total_chunks': len(chatbot_api.chunked_docs)
            })
        else:
            return jsonify({'error': 'Failed to initialize system with uploaded document'}), 500
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process', methods=['POST'])
def process_message():
    """Process message using selected strategy."""
    try:
        data = request.json
        strategy = data.get('strategy')
        message = data.get('message')
        options = data.get('options', {})
        
        if not strategy or not message:
            return jsonify({'error': 'Strategy and message are required'}), 400
        
        # Execute the strategy
        result = chatbot_api.execute_strategy(strategy, message, options)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about loaded documents."""
    if not chatbot_api.is_initialized:
        return jsonify({'error': 'No documents loaded'}), 400
    
    try:
        docs_info = []
        for i, doc in enumerate(chatbot_api.individual_documents[:10]):  # Limit to first 10
            docs_info.append({
                'id': i,
                'content_preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content,
                'metadata': doc.metadata,
                'content_length': len(doc.page_content)
            })
        
        return jsonify({
            'total_documents': len(chatbot_api.individual_documents),
            'total_chunks': len(chatbot_api.chunked_docs),
            'documents': docs_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Document Chatbot API Server...")
    logger.info("📋 Available endpoints:")
    logger.info("  - GET  /              : Chat interface")
    logger.info("  - POST /api/upload    : Upload document")
    logger.info("  - POST /api/process   : Process message")
    logger.info("  - GET  /api/status    : System status")
    logger.info("  - GET  /api/documents : Document info")
    
    app.run(debug=True, port=5050, host='0.0.0.0')