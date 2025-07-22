from ..abstracts.abstract_task_strategy import TaskStrategy
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from config.language_detect import returnlang
import re
import logging
from tqdm import tqdm


# Configure logging 
logger = logging.getLogger(__name__)

class ChattingStrategy(TaskStrategy):
    def __init__(self, llm, vector_store, embedder, top_k=5, return_sources=True):
        logger.info("💬 Initializing ChattingStrategy...")
        logger.info(f"⚙️ Configuration - top_k: {top_k}, return_sources: {return_sources}")
        
        self.llm = llm
        self.vector_store = vector_store
        self.top_k = top_k
        self.return_sources = return_sources
        
        logger.info("🔧 Setting embedder model on vector store...")
        self.vector_store.set_embedder_model(embedder)
        
        logger.info("🔗 Building processing chain...")
        self._build_chain()
        
        logger.info("✅ ChattingStrategy initialized successfully")

    def format_docs(self, docs):
        """Format retrieved documents for context."""
        logger.debug(f"📝 Formatting {len(docs)} documents for context...")
        
        try:
            formatted = "\n\n".join(
                f"[Source {i} | PDF {doc.metadata.get('pdf_id', '?')}]: {doc.page_content}"
                for i, doc in enumerate(docs, 1)
            )
            
            logger.debug(f"✅ Documents formatted - Total length: {len(formatted)} characters")
            return formatted
            
        except Exception as e:
            logger.error(f"❌ Error formatting documents: {str(e)}")
            raise
    def _build_chain(self):
        """Build the RAG processing chain."""
        logger.info("🔨 Building RAG chain components...")
        
        try:
            # Modified prompt template with language parameter
            prompt_template = """IMPORTANT: You must respond entirely in {detected_lang}. All sections, headers, and content must be in {detected_lang} language only.

            You are a helpful multilingual assistant. Use the following context to answer the question in {detected_lang}.

            Context:
            {context}

            Question: {question}

            Please provide a comprehensive answer based on the context above. You MUST respond entirely in {detected_lang} and follow this exact format structure:

            RESPONSE:
            [Your main answer here in {detected_lang}]

            REASONING:
            [Explain your reasoning and how you used the context in {detected_lang}]

            SOURCES:
            [List the source numbers you referenced, for example: 1, 3, 5]
            """
                    
            prompt = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question", "detected_lang"]
            )
            logger.debug("✅ Language-aware prompt template created")

            # Define retrieval function
            def retrieve_context(inputs):
                logger.debug(f"🔍 Retrieving context for question: '{inputs['question'][:50]}...'")
                docs = self.vector_store.get_relevant_documents(inputs["question"], top_k=self.top_k)
                logger.debug(f"📚 Retrieved {len(docs)} relevant documents")
                return self.format_docs(docs)

            # Language detection function for the chain
            def detect_language(inputs):
                detected_lang = returnlang(inputs["question"])
                logger.debug(f"🌐 Detected language: {detected_lang}")
                return detected_lang

            # Build the chain with language detection
            logger.debug("⛓️ Assembling processaing chain...")
            self.chain = ({
                    "context": RunnableLambda(retrieve_context),
                    "question": RunnablePassthrough(),
                    "detected_lang": RunnableLambda(detect_language)
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("✅ RAG chain built successfully with language detection")
            
        except Exception as e:
            logger.error(f"❌ Error building chain: {str(e)}")
            raise
    def parse_structured_response(self, response_text):
        """Parse the structured response from the LLM."""
        logger.debug("🔍 Parsing structured response...")
        
        try:
            # Clean response
            logger.debug("🧹 Cleaning response text...")
            with tqdm(total=3, desc="🔄 Cleaning text", unit="step", leave=False) as pbar:
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                pbar.update(1)
                
                cleaned_response = re.sub(r'<[^>]+>', '', cleaned_response)
                pbar.update(1)
                
                cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response.strip())
                pbar.update(1)

            logger.debug(f"✅ Response cleaned - Length: {len(cleaned_response)} characters")

            # Initialize sections
            sections = {'response': '', 'reasoning': '', 'sources': ''}
            current_section = None
            current_content = []

            # Parse sections
            lines = cleaned_response.split('\n')
            logger.debug(f"📄 Parsing {len(lines)} lines for sections...")
            
            for line in tqdm(lines, desc="📝 Parsing sections", unit="line", leave=False):
                line = line.strip()
                if line.upper().startswith('RESPONSE:'):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = 'response'
                    current_content = [line[9:].strip()]
                elif line.upper().startswith('REASONING:'):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = 'reasoning'
                    current_content = [line[10:].strip()]
                elif line.upper().startswith('SOURCES:'):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = 'sources'
                    current_content = [line[8:].strip()]
                elif current_section and line:
                    current_content.append(line)

            # Finalize current section
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()

            # Extract source IDs
            logger.debug("🔢 Extracting source IDs...")
            source_ids = [int(x) for x in re.findall(r'\d+', sections['sources'])] if sections['sources'] else []
            logger.debug(f"✅ Found {len(source_ids)} source IDs: {source_ids}")

            parsed_result = {
                'answer': sections['response'],
                'reasoning': sections['reasoning'],
                'sources': source_ids,
                'raw_response': cleaned_response
            }
            
            logger.debug("✅ Response parsing completed successfully")
            return parsed_result
            
        except Exception as e:
            logger.error(f"❌ Error parsing structured response: {str(e)}")
            raise

    def validate_input(self, question):
        """Validate that the question is a non-empty string."""
        logger.debug("🔍 Validating input question...")
        
        is_valid = isinstance(question, str) and len(question.strip()) > 0
        
        if is_valid:
            logger.debug(f"✅ Input validation passed - Question length: {len(question)} characters")
        else:
            logger.warning("❌ Input validation failed - Question is not a non-empty string")
            
        return is_valid

    def run(self, question):
        """Main method to run the chain and parse result."""
        logger.info(f"🚀 Starting chat processing for question: '{question[:50]}...'")
        
        try:
            # Validate input
            if not self.validate_input(question):
                logger.error("❌ Invalid input question provided")
                raise ValueError("Question must be a non-empty string")
            
            # Detect and log language before processing
            detected_lang = returnlang(question)
            logger.info(f"🌐 Processing question in detected language: {detected_lang}")
            
            # Process with chain - the chain will automatically detect language and pass it to prompt
            logger.info("⛓️ Invoking RAG chain...")
            with tqdm(total=1, desc="🤖 Processing query", unit="query") as pbar:
                response = self.chain.invoke({"question": question})
                pbar.update(1)
            
            logger.info("✅ Chain invocation completed")

            # Parse response
            logger.info("🔍 Parsing structured response...")
            parsed = self.parse_structured_response(response)
            logger.info("📊 Response parsed successfully")
            print(f"Parsed response: {parsed}")  

            # Retrieve source documents
            logger.info("📚 Retrieving source documents...")
            with tqdm(total=1, desc="📖 Getting sources", unit="retrieval") as pbar:
                source_docs = self.vector_store.get_relevant_documents(question, top_k=self.top_k)
                pbar.update(1)
            
            # Enhance parsed result with source information
            parsed['source_documents'] = source_docs
            parsed['source_texts'] = [doc.page_content for doc in source_docs]
            parsed['detected_language'] = detected_lang
            
            logger.info(f"✅ Chat processing completed - Retrieved {len(source_docs)} source documents")
            logger.info(f"📈 Answer length: {len(parsed['answer'])} characters")
            logger.info(f"🌐 Response language: {detected_lang}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"❌ Chat processing failed: {str(e)}")
            raise