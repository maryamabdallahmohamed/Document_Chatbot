import yaml
import logging
from tqdm import tqdm
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from src.abstracts.abstract_task_strategy import TaskStrategy
from config.language_detect import returnlang
from fuzzywuzzy import fuzz
from src.models.reranker import Reranker
from config.settings import FUZZY_THRESHOLD, DEFAULT_TOP_K

logger = logging.getLogger(__name__)

class SummarizationStrategy(TaskStrategy):
    def __init__(self, llm, template_file="config/prompts/summarization_prompts.yaml"):
        logger.info("📝 Initializing SummarizationStrategy...")
        self.llm = llm
        logger.info(f"📂 Loading templates from: {template_file}")
        self.load_templates(template_file)
        logger.info("✅ SummarizationStrategy initialized successfully")
    
    def load_templates(self, template_file):
        """Load templates from YAML file."""
        logger.info(f"🔄 Loading YAML templates from {template_file}...")
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self.summary_templates = data.get('summary_templates', {})
            self.overview_templates = data.get('overview_templates', {})
            
            summary_count = len(self.summary_templates)
            overview_count = len(self.overview_templates)
            logger.info(f"✅ Templates loaded: {summary_count} summary templates, {overview_count} overview templates")
            
        except FileNotFoundError:
            logger.error(f"❌ Template file not found: {template_file}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ Error parsing YAML file: {e}")
            raise
    

    def validate_input(self, document):
        """Validate that the document is a non-empty string."""
        logger.debug("🔍 Validating input document...")
        is_valid = isinstance(document, str) and len(document.strip()) > 0
        
        if is_valid:
            logger.debug(f"✅ Input validation passed - Document length: {len(document)} characters")
        else:
            logger.warning("❌ Input validation failed - Document is not a non-empty string")
            
        return is_valid

    def run(self, document, length="medium", verbose=False, overview_level=None):
        """
        Summarize the given document with customizable length and optional reasoning,
        or create an overview at specified level and length.
        """
        logger.info(f"🚀 Starting summarization task - Length: {length}, Verbose: {verbose}, Overview Level: {overview_level}")
        
        if not self.validate_input(document):
            logger.error("❌ Invalid input document provided")
            
            raise ValueError("Document must be a non-empty string")
        
        # Detect language from document
        detected_lang = returnlang(document)
            
        try:
            if overview_level:
                logger.info(f"📊 Creating overview at level: {overview_level}")
                return self._create_overview(document, overview_level, length, verbose, detected_lang)
            else:
                logger.info("📄 Creating regular summary")
                return self._create_summary(document, length, verbose, detected_lang)
                
        except Exception as e:
            logger.error(f"❌ Summarization task failed: {str(e)}")
            raise
    
    def _create_overview(self, document, overview_level, length, verbose=False, detected_lang="English"):
        """Create an overview of the document at the specified level and length."""
        logger.info(f"🔧 Preparing overview template - Level: {overview_level}, Length: {length}, Verbose: {verbose}, Language: {detected_lang}")
        
        try:
            template_type = "with_reasoning" if verbose else "base"
            prompt_text = self.overview_templates[overview_level][length][template_type]
            logger.debug(f"📋 Selected template type: {template_type}")
            
            # Add language instruction to the prompt
            language_instruction = f"IMPORTANT: You must respond entirely in {detected_lang}. All content, headers, and explanations must be in {detected_lang} language only.\n\n"
            enhanced_prompt_text = language_instruction + prompt_text
            
            with tqdm(total=2, desc="🔄 Creating overview", unit="step") as pbar:
                logger.info("🛠️ Formatting prompt template...")
                prompt_template = ChatPromptTemplate.from_messages([("system", enhanced_prompt_text)])
                formatted_prompt = prompt_template.format(context=document)
                pbar.update(1)
                
                logger.info("🤖 Invoking LLM for overview generation...")
                result = self.llm.invoke(formatted_prompt)
                pbar.update(1)
            
            logger.info(f"✅ Overview created successfully in {detected_lang}")
            print(result)
            return result
            
        except KeyError as e:
            logger.error(f"❌ Template not found for overview_level={overview_level}, length={length}, template_type={template_type}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating overview: {str(e)}")
            raise
    
    def _create_summary(self, document, length, verbose, detected_lang="English"):
        """Create a regular summary of the document."""
        logger.info(f"🔧 Preparing summary template - Length: {length}, Verbose: {verbose}, Language: {detected_lang}")
        
        try:
            template_type = "with_reasoning" if verbose else "base"
            prompt_text = self.summary_templates[length][template_type]
            logger.debug(f"📋 Selected template type: {template_type}")
            
            # Add language instruction to the prompt
            language_instruction = f"IMPORTANT: You must respond entirely in {detected_lang}. All content, headers, and explanations must be in {detected_lang} language only.\n\n"
            enhanced_prompt_text = language_instruction + prompt_text
            
            with tqdm(total=2, desc="🔄 Creating summary", unit="step") as pbar:
                logger.info("🛠️ Formatting prompt template...")
                prompt_template = ChatPromptTemplate.from_messages([("system", enhanced_prompt_text)])
                formatted_prompt = prompt_template.format(context=document)
                pbar.update(1)
                
                logger.info("🤖 Invoking LLM for summary generation...")
                result = self.llm.invoke(formatted_prompt)
                pbar.update(1)
            
            logger.info(f"✅ Summary created successfully in {detected_lang}")
            print(result)
            return result
            
        except KeyError as e:
            logger.error(f"❌ Template not found for length={length}, template_type={template_type}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating summary: {str(e)}")
            raise




class Summarization_Rag_Strategy(TaskStrategy):
    def __init__(self, llm, retriever, fuzzy_threshold: int = None, top_k: int = None):
        logger.info("🔍 Initializing Summarization_Rag_Strategy...")
        if fuzzy_threshold is None:
            fuzzy_threshold = FUZZY_THRESHOLD
        if top_k is None:
            top_k = DEFAULT_TOP_K
        
        self.llm = llm
        self.retriever = retriever
        self.fuzzy_threshold = fuzzy_threshold
        self.top_k = top_k
        
        # Import and initialize reranker

        self.reranker = Reranker()
        logger.info("✅ Reranker initialized successfully")
        logger.info("✅ Summarization_Rag_Strategy initialized successfully")

    def _create_language_aware_prompt(self, detected_lang):
        """Create a language-aware RAG prompt template."""
        logger.info(f"📋 Setting up RAG prompt template for {detected_lang}...")
        
        template = f"""
        IMPORTANT: You must respond entirely in {detected_lang}. All content, headers, and explanations must be in {detected_lang} language only.

        You are a helpful assistant analyzing content related to a specific keyword.

        The user searched for the keyword: "{{keyword}}"

        Based on the following document excerpts (starting with the keyword-matched chunk and followed by semantically similar content), generate a structured summary in {detected_lang}.

        Only use the provided content—do not include prior knowledge or assumptions.

        == Document Excerpts ==
        {{context}}

        == Summary ==
        **Keyword Context:** [Explain how the keyword "{{keyword}}" appears in the content in {detected_lang}.]

        **Main Topic:** [Summarize the general theme of the retrieved content in {detected_lang}.]

        **Key Points:**
        - [Most relevant insight #1 in {detected_lang}]
        - [Relevant insight #2 in {detected_lang}]
        - [Relevant insight #3 in {detected_lang}]

        **Supporting Details:** [Specific numbers, quotes, or facts in {detected_lang}.]

        **Conclusion:** [Key implication or recommendation in {detected_lang}.]
        """
        
        return PromptTemplate(
            input_variables=["keyword", "context"],
            template=template
        )

    def _fuzzy_search_chunks(self, keyword, all_chunks):
        """
        Search for chunks containing the keyword using fuzzy logic.
        Returns the best matching chunk or None if no match above threshold.
        """
        logger.info(f"🔍 Performing fuzzy search for keyword: '{keyword}'")
        
        best_match = None
        best_score = 0
        
        chunk_texts = []
        for i, chunk in enumerate(all_chunks):
            chunk_texts.append((i, chunk.page_content))
        
        # Search through chunk contents
        for idx, content in tqdm(chunk_texts, desc="🔍 Fuzzy searching", unit="chunk"):
            # Split content into words for better matching
            words = content.lower().split()
            
            # Check fuzzy match against individual words
            for word in words:
                score = fuzz.ratio(keyword.lower(), word)
                if score > best_score and score >= self.fuzzy_threshold:
                    best_score = score
                    best_match = all_chunks[idx]
            
            # Also check fuzzy match against phrases (sliding window)
            content_lower = content.lower()
            score = fuzz.partial_ratio(keyword.lower(), content_lower)
            if score > best_score and score >= self.fuzzy_threshold:
                best_score = score
                best_match = all_chunks[idx]
        
        if best_match:
            logger.info(f"✅ Found fuzzy match with score {best_score}")
            logger.debug(f"📄 Matched chunk preview: {best_match.page_content[:100]}...")
        else:
            logger.warning(f"❌ No fuzzy match found above threshold {self.fuzzy_threshold}")
        
        return best_match

    def _get_similar_chunks(self, seed_chunk, top_k):
        """
        Retrieve top k chunks semantically similar to the seed chunk using FAISS.
        Applies reranking if available.
        """
        logger.info(f"🔎 Finding {top_k} chunks similar to seed chunk...")
        
        try:
            # Retrieve more chunks for reranking if reranker is available
            retrieve_k = top_k * 2 if self.reranker else top_k + 1
            
            # Use the seed chunk's content as query for similarity search
            similar_chunks = self.retriever.get_relevant_documents(
                query=seed_chunk.page_content,
                top_k=retrieve_k
            )
            
            # Remove the seed chunk from results if present (avoid duplication)
            filtered_chunks = []
            seed_content = seed_chunk.page_content.strip()
            
            for chunk in similar_chunks:
                chunk_content = chunk.page_content.strip()
                # Skip if it's the exact same content as seed chunk
                if chunk_content != seed_content:
                    filtered_chunks.append(chunk)
                elif len(filtered_chunks) == 0:
                    # Keep the first occurrence (in case seed chunk is the most similar)
                    filtered_chunks.append(chunk)
            
            # Apply reranking if we have chunks to rerank
            if len(filtered_chunks) > 1:
                logger.info("🔄 Applying reranking to retrieved chunks...")
                filtered_chunks = self.reranker.rerank_chunks(seed_chunk.page_content, filtered_chunks)
                logger.info("✅ Reranking completed successfully")
            
            # Ensure we have the right number of chunks
            final_chunks = filtered_chunks[:top_k]
            
            # If we don't have enough chunks, add the seed chunk back
            if len(final_chunks) < top_k and seed_chunk not in final_chunks:
                # Create a copy of seed chunk with similarity metadata
                seed_copy = Document(
                    page_content=seed_chunk.page_content,
                    metadata={**seed_chunk.metadata, "similarity": 1.0, "is_seed": True}
                )
                final_chunks.insert(0, seed_copy)
                final_chunks = final_chunks[:top_k]
            
            logger.info(f"✅ Retrieved {len(final_chunks)} similar chunks")
            return final_chunks
            
        except Exception as e:
            logger.error(f"❌ Error retrieving similar chunks: {str(e)}")
            # Fallback: return just the seed chunk with enhanced metadata
            fallback_chunk = Document(
                page_content=seed_chunk.page_content,
                metadata={**seed_chunk.metadata, "similarity": 1.0, "is_seed": True, "fallback": True}
            )
            return [fallback_chunk]

    def _get_all_chunks(self):
        """
        Retrieve all available chunks from the FAISS vector store.
        """
        logger.info("📚 Retrieving all available chunks...")
        
        try:
            # Method 1: Direct access to documents attribute (for Fais_VS)
            if hasattr(self.retriever, 'documents') and self.retriever.documents:
                logger.info(f"✅ Found {len(self.retriever.documents)} documents in Fais_VS")
                return self.retriever.documents
            
            # Method 2: Access through docstore (for Fais_VS)
            elif hasattr(self.retriever, 'docstore') and self.retriever.docstore:
                logger.info(f"✅ Found {len(self.retriever.docstore)} documents in docstore")
                return list(self.retriever.docstore.values())
            
            # Method 3: Reconstruct from chunks_dict (backward compatibility)
            elif hasattr(self.retriever, 'chunks_dict') and self.retriever.chunks_dict:
                logger.info(f"✅ Found {len(self.retriever.chunks_dict)} chunks in chunks_dict")
                documents = []
                for idx, text in self.retriever.chunks_dict.items():
                    doc = Document(
                        page_content=text,
                        metadata={"chunk_id": idx, "source": "chunks_dict"}
                    )
                    documents.append(doc)
                return documents
            
            # Method 4: Generic fallback for other retriever types
            elif hasattr(self.retriever, 'get_relevant_documents'):
                # Use a broad query to get diverse results
                logger.info("⚠️ Falling back to broad search method")
                chunks = self.retriever.get_relevant_documents("", k=1000)  # Adjust k as needed
                return chunks
            
            logger.error("❌ Cannot retrieve all chunks - unsupported retriever type")
            raise ValueError("Unable to retrieve all chunks from retriever. Ensure your retriever is a Fais_VS instance or has documents/docstore attributes.")
            
        except Exception as e:
            logger.error(f"❌ Error retrieving all chunks: {str(e)}")
            raise

    def validate_input(self, keyword):
        """Validate that the input keyword is valid."""
        logger.debug("🔍 Validating input keyword...")
        is_valid = isinstance(keyword, str) and len(keyword.strip()) > 0
        
        if is_valid:
            logger.debug(f"✅ Input validation passed - keyword: '{keyword}'")
        else:
            logger.warning("❌ Input validation failed - Invalid keyword")
            
        return is_valid

    def run(self, keyword):
        """
        Main execution method:
        1. Fuzzy search for keyword in chunks
        2. Find semantically similar chunks to the keyword-matched chunk
        3. Apply reranking for better relevance (if available)
        4. Generate summary using LLM
        """
        reranker_status = "✅ enabled"
        logger.info(f"🚀 Starting RAG for keyword: '{keyword}' (reranker: {reranker_status})")
        
        # Validate input
        if not self.validate_input(keyword):
            raise ValueError("Invalid keyword provided")
        
        # Detect language from keyword
        detected_lang = returnlang(keyword)
        logger.info(f"🌐 Keyword language detected as: {detected_lang}")
        
        try:
            # Step 1: Get all available chunks
            logger.info("📚 Retrieving all chunks for fuzzy search...")
            all_chunks = self._get_all_chunks()
            logger.info(f"📊 Total chunks available: {len(all_chunks)}")
            
            # Step 2: Fuzzy search for keyword
            seed_chunk = self._fuzzy_search_chunks(keyword, all_chunks)
            
            if not seed_chunk:
                logger.error(f"❌ No chunks found matching keyword '{keyword}'")
                raise ValueError(f"No chunks found matching keyword '{keyword}' above fuzzy threshold {self.fuzzy_threshold}")
            
            # Step 3: Get semantically similar chunks (with optional reranking)
            similar_chunks = self._get_similar_chunks(seed_chunk, self.top_k)
            
            # Step 4: Combine chunks for context
            logger.info("🔗 Preparing context from matched chunks...")
            combined_chunks = []
            
            # Add the seed chunk first (keyword-matched)
            seed_text = f"[KEYWORD MATCH - Page {seed_chunk.metadata.get('page', 'N/A')}]:\n{seed_chunk.page_content}"
            combined_chunks.append(seed_text)
            
            # Add similar chunks with reranking info if available
            for i, chunk in enumerate(similar_chunks[1:], 1):  # Skip first if it's the seed chunk
                rerank_info = ""
                if chunk.metadata.get("rerank_score") is not None:
                    rerank_info = f" (Rerank: {chunk.metadata['rerank_score']:.3f})"
                
                chunk_text = f"[SIMILAR CHUNK #{i} - Page {chunk.metadata.get('page', 'N/A')}{rerank_info}]:\n{chunk.page_content}"
                combined_chunks.append(chunk_text)
            
            combined_text = "\n\n".join(combined_chunks)
            logger.info(f"📝 Combined context length: {len(combined_text)} characters")
            
            # Step 5: Create language-aware prompt
            language_prompt = self._create_language_aware_prompt(detected_lang)
                        
            # Step 6: Generate summary
            logger.info(f"🤖 Generating keyword-based summary in {detected_lang}...")
            formatted_prompt = language_prompt.format(
                keyword=keyword,
                context=combined_text
            )

            with tqdm(desc="🤖 Generating summary", bar_format="{desc}... {elapsed}"):
                result = self.llm.invoke(formatted_prompt)
            logger.info(f"✅ RAG completed successfully for keyword '{keyword}'")
            print(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG failed: {str(e)}")
            raise
