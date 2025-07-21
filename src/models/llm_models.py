
from langchain_ollama import OllamaLLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from tqdm import tqdm
from ..abstracts.abstract_llm import BaseLLM
logger = logging.getLogger(__name__)

class OLLAMA_LLM(BaseLLM):
    def __init__(self, model_name, cache_folder):
        logger.info(f"🤖 Initializing OLLAMA_LLM with model: {model_name}")
        logger.info(f"📁 Cache folder: {cache_folder}")
        super().__init__(model_name, cache_folder)
        logger.info("✅ OLLAMA_LLM initialized successfully")

    def load_model(self):
        logger.info(f"🔥 Loading OLLAMA model: {self.model_name}")
        logger.info("⚙️ Configuration - Temperature: 0.3, Context: 4096")
        
        try:
            with tqdm(total=1, desc="🤖 Loading OLLAMA", unit="model") as pbar:
                model = OllamaLLM(
                    model=self.model_name, 
                    temperature=0.3, 
                    num_ctx=4096
                )
                pbar.update(1)
            
            logger.info("✅ OLLAMA model loaded successfully")
            logger.info(f"📊 Model details - Name: {self.model_name}, Temperature: 0.3, Context: 4096")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load OLLAMA model: {str(e)}")
            raise


class Hugging_Face_LLM(BaseLLM):
    def __init__(self, model_name, cache_folder):
        logger.info(f"🤗 Initializing Hugging Face LLM with model: {model_name}")
        logger.info(f"📁 Cache folder: {cache_folder}")
        super().__init__(model_name, cache_folder)
        logger.info("✅ Hugging Face LLM initialized successfully")

    def load_model(self):
        logger.info(f"🔥 Loading Hugging Face model: {self.model_name}")
        logger.info(f"💾 Cache directory: {self.cache_folder}")
        logger.info(f"🖥️ Target device: {getattr(self, 'device', 'auto')}")
        
        try:
            # Determine device and dtype
            device = getattr(self, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            logger.info(f"🎯 Using device: {device}, dtype: {dtype}")
            
            # Load tokenizer
            logger.info("📝 Loading tokenizer...")
            with tqdm(total=1, desc="📝 Loading tokenizer", unit="component") as pbar:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_folder
                )
                pbar.update(1)
            
            logger.info("✅ Tokenizer loaded successfully")
            logger.info(f"📊 Tokenizer vocab size: {len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 'Unknown'}")
            

            logger.info("🧠 Loading language model...")
            with tqdm(total=1, desc="🧠 Loading model", unit="component") as pbar:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_folder,
                    torch_dtype=dtype,
                    device_map="auto"  
                )
                pbar.update(1)
            
            logger.info("✅ Language model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Hugging Face model: {str(e)}")
            if "out of memory" in str(e).lower():
                logger.error("💡 Suggestion: Try using a smaller model or reduce batch size")
            elif "connection" in str(e).lower():
                logger.error("💡 Suggestion: Check internet connection for model download")
            raise