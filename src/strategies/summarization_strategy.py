import yaml
import logging
from tqdm import tqdm
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from src.abstracts.abstract_task_strategy import TaskStrategy


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
            
        try:
            if overview_level:
                logger.info(f"📊 Creating overview at level: {overview_level}")
                return self._create_overview(document, overview_level, length, verbose)
            else:
                logger.info("📄 Creating regular summary")
                return self._create_summary(document, length, verbose)
                
        except Exception as e:
            logger.error(f"❌ Summarization task failed: {str(e)}")
            raise
    
    def _create_overview(self, document, overview_level, length, verbose=False):
        """Create an overview of the document at the specified level and length."""
        logger.info(f"🔧 Preparing overview template - Level: {overview_level}, Length: {length}, Verbose: {verbose}")
        
        try:
            template_type = "with_reasoning" if verbose else "base"
            prompt_text = self.overview_templates[overview_level][length][template_type]
            logger.debug(f"📋 Selected template type: {template_type}")
            
            with tqdm(total=2, desc="🔄 Creating overview", unit="step") as pbar:
                logger.info("🛠️ Formatting prompt template...")
                prompt_template = ChatPromptTemplate.from_messages([("system", prompt_text)])
                formatted_prompt = prompt_template.format(context=document)
                pbar.update(1)
                
                logger.info("🤖 Invoking LLM for overview generation...")
                result = self.llm.invoke(formatted_prompt)
                pbar.update(1)
            
            logger.info("✅ Overview created successfully")
            print(result)
            return result
            
        except KeyError as e:
            logger.error(f"❌ Template not found for overview_level={overview_level}, length={length}, template_type={template_type}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating overview: {str(e)}")
            raise
    
    def _create_summary(self, document, length, verbose):
        """Create a regular summary of the document."""
        logger.info(f"🔧 Preparing summary template - Length: {length}, Verbose: {verbose}")
        
        try:
            template_type = "with_reasoning" if verbose else "base"
            prompt_text = self.summary_templates[length][template_type]
            logger.debug(f"📋 Selected template type: {template_type}")
            
            with tqdm(total=2, desc="🔄 Creating summary", unit="step") as pbar:
                logger.info("🛠️ Formatting prompt template...")
                prompt_template = ChatPromptTemplate.from_messages([("system", prompt_text)])
                formatted_prompt = prompt_template.format(context=document)
                pbar.update(1)
                
                logger.info("🤖 Invoking LLM for summary generation...")
                result = self.llm.invoke(formatted_prompt)
                pbar.update(1)
            
            logger.info("✅ Summary created successfully")
            print(result)
            return result
            
        except KeyError as e:
            logger.error(f"❌ Template not found for length={length}, template_type={template_type}")
            raise
        except Exception as e:
            logger.error(f"❌ Error creating summary: {str(e)}")
            raise

class Summarization_Rag_Strategy(TaskStrategy):
    def __init__(self, llm, retriever):
        logger.info("🔍 Initializing Summarization_Rag_Strategy...")
        self.llm = llm
        self.retriever = retriever
        
        logger.info("📋 Setting up RAG prompt template...")
        self.prompt = PromptTemplate(
            input_variables=["user_prompt", "context"],
            template="""
            You are a helpful assistant.

            The user is interested in the topic: "{user_prompt}"

            Based on the following document excerpts, generate a structured summary.

            Only use the provided content—do not include prior knowledge or assumptions.

            == Document Excerpts ==
            {context}

            == Summary ==
            **Main Topic:** [Summarize the general theme of the retrieved content.]

            **Key Points:**
            - [Most relevant insight #1]
            - [Relevant insight #2]
            - [Relevant insight #3]

            **Supporting Details:** [Specific numbers, quotes, or facts.]

            **Conclusion:** [Key implication or recommendation.]
            """
        )
        logger.info("✅ Summarization_Rag_Strategy initialized successfully")

    def validate_input(self, documents):
        """Validate that the input is a non-empty list of Document objects."""
        logger.debug("🔍 Validating input documents...")
        is_valid = isinstance(documents, list) and all(isinstance(doc, Document) for doc in documents)
        
        if is_valid:
            logger.debug(f"✅ Input validation passed - {len(documents)} documents provided")
        else:
            logger.warning("❌ Input validation failed - Not a valid list of Document objects")
            
        return is_valid

    def run(self, prompt):
        """Retrieve and summarize relevant chunks."""
        logger.info(f"🔍 Starting RAG summarization for prompt: '{prompt[:50]}...' ")
        
        try:
            # Retrieve similar chunks
            logger.info("🔎 Retrieving relevant documents...")
            with tqdm(total=1, desc="🔍 Retrieving docs", unit="query") as pbar:
                similar_chunks = self.retriever.get_relevant_documents(prompt)
                pbar.update(1)
            
            logger.info(f"📚 Retrieved {len(similar_chunks)} similar chunks")

            # Filter positively correlated chunks
            logger.info("🔬 Filtering chunks by similarity threshold...")
            positively_correlated = [
                chunk for chunk in tqdm(similar_chunks, desc="🔬 Filtering chunks", unit="chunk")
                if chunk.metadata.get('similarity', 0) > 0.1
            ]

            logger.info(f"✅ {len(positively_correlated)} chunks passed similarity threshold (> 0.1)")

            if not positively_correlated:
                logger.error("❌ No chunks above similarity threshold")
                raise ValueError("No chunks above similarity threshold.")

            # Combine text from filtered chunks
            logger.info("🔗 Combining text from relevant chunks...")
            combined_chunks = []
            
            for doc in tqdm(positively_correlated, desc="🔗 Combining chunks", unit="chunk"):
                chunk_text = f"[Chunk from page {doc.metadata.get('page', 'N/A')}]:\n{doc.page_content}"
                combined_chunks.append(chunk_text)
            
            combined_text = "\n\n".join(combined_chunks)
            logger.info(f"📝 Combined text length: {len(combined_text)} characters")

            # Generate summary using LLM
            logger.info("🤖 Generating RAG summary...")
            with tqdm(total=2, desc="🤖 Generating summary", unit="step") as pbar:
                logger.info("🛠️ Formatting RAG prompt...")
                formatted_prompt = self.prompt.format(user_prompt=prompt, context=combined_text)
                pbar.update(1)
                
                logger.info("🔥 Invoking LLM for RAG summary...")
                result = self.llm.invoke(formatted_prompt)
                pbar.update(1)

            logger.info("✅ RAG summarization completed successfully")
            print(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG summarization failed: {str(e)}")
            raise