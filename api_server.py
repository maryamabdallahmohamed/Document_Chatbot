from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import os
import sys
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback
from flask import Flask, send_from_directory
from threading import Lock
import threading
from src.core.chat_history_manager import ConversationHistoryManager

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
from config.settings import (
    DEFAULT_EMBEDDING_MODEL, DEFAULT_BATCH_SIZE, OLLAMA_MODELS,
    API_HOST, API_PORT, API_DEBUG, UPLOAD_FOLDER, ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH, LOG_LEVEL, LLM_CACHE_DIR
)

app = Flask(__name__, 
           template_folder='frontend',
           static_folder='frontend')
CORS(app)

# Configuration
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("user_uploads", exist_ok=True)  # Directory for user-specific uploads
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DocumentChatbotAPI:
    def __init__(self, user_id=None):
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
        self.history_manager = ConversationHistoryManager()
        self.user_id = user_id or "default"
        self.lock = Lock()  # Thread lock for this user's operations
        
    def setup_logging(self):
        """Configure logging for the pipeline."""
        return logging.getLogger(__name__)

    def load_and_process_documents(self, files_paths):
        """Load and process documents into chunks (adapted from your bot.py)."""
        logger.info(f"📁 User {self.user_id}: Loading documents...")
        paths = [files_paths]
        docs = JSONPreprocessor()

        data = docs.process_documents_from_files(paths)
        logger.info(f"✅ User {self.user_id}: Loaded {len(data)} documents")

        logger.info(f"📄 User {self.user_id}: Creating individual documents...")
        individual_documents = [
            Document(page_content=pdf.page_content, metadata={"pdf_id": i, "user_id": self.user_id})
            for i, pdf in enumerate(data)
            if pdf.page_content
        ]
        logger.info(f"✅ User {self.user_id}: Created {len(individual_documents)} individual documents")

        logger.info(f"✂️ User {self.user_id}: Chunking documents...") 
        chunked_docs = docs.chunk_documents(individual_documents)
        logger.info(f"✅ User {self.user_id}: Document chunking completed - {len(chunked_docs)} chunks created")

        return chunked_docs, individual_documents

    def initialize_models(self):
        """Initialize embedder and LLM models."""
        with self.lock:
            logger.info(f"🧠 User {self.user_id}: Initializing multilingual embedder model...")
            self.multilingual_embedder = MultilingualEmbedder(
                model_name=DEFAULT_EMBEDDING_MODEL, 
                batch_size=DEFAULT_BATCH_SIZE
            )
            logger.info(f"✅ User {self.user_id}: Embedder model loaded")

            logger.info(f"🤖 User {self.user_id}: Loading OLLAMA LLM model...")
            # Create user-specific cache directory
            user_cache_dir = f'{LLM_CACHE_DIR}/user_{self.user_id[:8]}'
            os.makedirs(user_cache_dir, exist_ok=True)
            self.llm = OLLAMA_LLM(OLLAMA_MODELS['qwen3'], f'{user_cache_dir}/qwen3_cache').load_model()
            logger.info(f"✅ User {self.user_id}: LLM model loaded")

    def create_vector_store(self):
        """Create and populate vector store with document embeddings."""
        with self.lock:
            logger.info(f"🗄️ User {self.user_id}: Creating vector store...")
            self.vector_store = Fais_VS()
            self.vector_store.set_embedder_model(self.multilingual_embedder)
            self.vector_store.create_vector_store(self.chunked_docs)
            logger.info(f"✅ User {self.user_id}: Vector store created with {len(self.chunked_docs)} embeddings")

    def initialize_strategies(self):
        """Initialize all processing strategies."""
        with self.lock:
            logger.info(f"⚙️ User {self.user_id}: Initializing processing strategies...")
            
            self.chatting_strategy = ChattingStrategy(self.llm, self.vector_store, self.multilingual_embedder)
            self.summarization_strategy = SummarizationStrategy(self.llm)
            self.question_strategy = QuestionStrategy(self.llm)
            self.rag_summary = Summarization_Rag_Strategy(self.llm, self.vector_store)
            self.processor = TaskProcessor()
            
            logger.info(f"✅ User {self.user_id}: All strategies initialized")

    def initialize_system(self, document_path):
        """Initialize the entire system with a document."""
        try:
            logger.info(f"🚀 User {self.user_id}: Initializing document processing system...")
            
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
            logger.info(f"✅ User {self.user_id}: System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ User {self.user_id}: System initialization failed: {str(e)}")
            return False

    def execute_strategy(self, strategy_name, message, options=None, conversation_id="default"):
        """Execute a strategy with given parameters."""
        if not self.is_initialized:
            return {"error": "System not initialized. Please upload a document first."}
        
        try:
            with self.lock:
                if options is None:
                    options = {}
                    
                logger.info(f"🔄 User {self.user_id}: Executing {strategy_name} for conversation {conversation_id}")
                
                if strategy_name == 'chatting_strategy':
                    self.processor.strategy = self.chatting_strategy
                    result = self.processor.execute_task(message, conversation_id)
                    result = result['answer']
                    
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
                    result = result['qa_output']
                    
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
                
                logger.info(f"✅ User {self.user_id}: Strategy {strategy_name} completed successfully")
                
                # Return raw result without any HTML formatting
                return {"response": str(result)}
            
        except Exception as e:
            logger.error(f"❌ User {self.user_id}: Strategy execution error: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Strategy execution failed: {str(e)}"}

# Global dictionaries to store user instances (maintaining original API structure)
user_instances = {}
user_instances_lock = Lock()
# Also maintain a global instance for backward compatibility
chatbot_api = None

def get_user_instance(user_id=None):
    """Get or create a user instance."""
    if user_id is None:
        # Backward compatibility - use global instance
        global chatbot_api
        if chatbot_api is None:
            chatbot_api = DocumentChatbotAPI("default")
        return chatbot_api
    
    with user_instances_lock:
        if user_id not in user_instances:
            logger.info(f"🆕 Creating new instance for user {user_id}")
            user_instances[user_id] = DocumentChatbotAPI(user_id)
        return user_instances[user_id]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

# ORIGINAL API ENDPOINTS (maintaining exact same structure)
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status - original endpoint for backward compatibility."""
    chatbot_api = get_user_instance()
    return jsonify({
        'initialized': chatbot_api.is_initialized,
        'document_loaded': chatbot_api.current_document_path is not None,
        'current_document': chatbot_api.current_document_path,
        'total_documents': len(chatbot_api.individual_documents) if chatbot_api.individual_documents else 0,
        'total_chunks': len(chatbot_api.chunked_docs) if chatbot_api.chunked_docs else 0
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and system initialization - original endpoint."""
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
        chatbot_api = get_user_instance()
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
    """Process message using selected strategy - original endpoint."""
    try:
        data = request.json
        strategy = data.get('strategy')
        message = data.get('message')
        options = data.get('options', {})
        conversation_id = data.get('conversation_id', 'default')
        
        if not strategy or not message:
            return jsonify({'error': 'Strategy and message are required'}), 400
        
        # Execute the strategy
        chatbot_api = get_user_instance()
        result = chatbot_api.execute_strategy(strategy, message, options, conversation_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get information about loaded documents - original endpoint."""
    chatbot_api = get_user_instance()
    
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

# NEW PARALLEL USER ENDPOINTS (for multi-user support)
@app.route('/api/generate-user-id', methods=['POST'])
def generate_user_id():
    """Generate a new unique user ID."""
    user_id = str(uuid.uuid4())
    logger.info(f"🆔 Generated new user ID: {user_id}")
    return jsonify({'user_id': user_id})

@app.route('/api/status/<user_id>', methods=['GET'])
def get_user_status(user_id):
    """Get system status for a specific user."""
    chatbot_api = get_user_instance(user_id)
    return jsonify({
        'initialized': chatbot_api.is_initialized,
        'document_loaded': chatbot_api.current_document_path is not None,
        'current_document': chatbot_api.current_document_path,
        'total_documents': len(chatbot_api.individual_documents) if chatbot_api.individual_documents else 0,
        'total_chunks': len(chatbot_api.chunked_docs) if chatbot_api.chunked_docs else 0,
        'user_id': user_id
    })

@app.route('/api/upload/<user_id>', methods=['POST'])
def upload_user_file(user_id):
    """Handle file upload and system initialization for a specific user."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload JSON, TXT, or PDF files.'}), 400
        
        # Create user-specific upload directory
        user_upload_dir = os.path.join("user_uploads", user_id)
        os.makedirs(user_upload_dir, exist_ok=True)
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(user_upload_dir, filename)
        file.save(filepath)

        logger.info(f"📁 User {user_id}: File uploaded: {filepath}")
        
        # Get user instance and initialize the system with the new document
        chatbot_api = get_user_instance(user_id)
        
        # Initialize in a separate thread to avoid blocking
        def initialize_async():
            success = chatbot_api.initialize_system(filepath)
            if success:
                logger.info(f"✅ User {user_id}: System initialization completed")
            else:
                logger.error(f"❌ User {user_id}: System initialization failed")
        
        # Start initialization in background
        init_thread = threading.Thread(target=initialize_async)
        init_thread.daemon = True
        init_thread.start()
        
        return jsonify({
            'message': f'File {filename} uploaded successfully. System initialization started.',
            'filename': filename,
            'filepath': filepath,
            'user_id': user_id,
            'status': 'initializing'
        })
        
    except Exception as e:
        logger.error(f"❌ User {user_id}: Upload error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/process/<user_id>', methods=['POST'])
def process_user_message(user_id):
    """Process message using selected strategy for a specific user."""
    try:
        data = request.json
        strategy = data.get('strategy')
        message = data.get('message')
        options = data.get('options', {})
        conversation_id = data.get('conversation_id', 'default')
        
        if not strategy or not message:
            return jsonify({'error': 'Strategy and message are required'}), 400
        
        # Get user instance
        chatbot_api = get_user_instance(user_id)
        
        # Execute the strategy
        result = chatbot_api.execute_strategy(strategy, message, options, conversation_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ User {user_id}: Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/documents/<user_id>', methods=['GET'])
def get_user_documents(user_id):
    """Get information about loaded documents for a specific user."""
    chatbot_api = get_user_instance(user_id)
    
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
            'documents': docs_info,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup/<user_id>', methods=['POST'])
def cleanup_user(user_id):
    """Clean up a user's session and resources."""
    try:
        with user_instances_lock:
            if user_id in user_instances:
                logger.info(f"🗑️ Cleaning up instance for user {user_id}")
                del user_instances[user_id]
        
        # Optionally clean up user files (uncomment if desired)
        # import shutil
        # user_upload_dir = os.path.join("user_uploads", user_id)
        # if os.path.exists(user_upload_dir):
        #     shutil.rmtree(user_upload_dir)
        
        return jsonify({'message': f'User {user_id} cleaned up successfully'})
    except Exception as e:
        logger.error(f"❌ Cleanup error for user {user_id}: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.route('/api/active-users', methods=['GET'])
def get_active_users():
    """Get list of active users (for admin purposes)."""
    with user_instances_lock:
        active_users = list(user_instances.keys())
    return jsonify({
        'active_users': active_users,
        'count': len(active_users)
    })

# Static file serving
@app.route('/')
def serve_index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('frontend', filename)

if __name__ == '__main__':
    logger.info("🚀 Starting Multi-User Document Chatbot API Server...")
    logger.info("📋 Available endpoints:")
    logger.info("  - GET  /                         : Chat interface")
    logger.info("  - POST /api/upload               : Upload document (original)")
    logger.info("  - POST /api/process              : Process message (original)")
    logger.info("  - GET  /api/status               : System status (original)")
    logger.info("  - GET  /api/documents            : Document info (original)")
    logger.info("  - POST /api/generate-user-id     : Generate new user ID")
    logger.info("  - POST /api/upload/<user_id>     : Upload document for user")
    logger.info("  - POST /api/process/<user_id>    : Process message for user")
    logger.info("  - GET  /api/status/<user_id>     : System status for user")
    logger.info("  - GET  /api/documents/<user_id>  : Document info for user")
    logger.info("  - POST /api/cleanup/<user_id>    : Cleanup user session")
    logger.info("  - GET  /api/active-users         : List active users")
    
    app.run(debug=API_DEBUG, port=API_PORT, host=API_HOST, threaded=True)