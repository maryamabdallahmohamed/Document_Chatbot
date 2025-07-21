from ..abstracts.abstract_task_strategy import TaskStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import re
import logging
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class QuestionStrategy(TaskStrategy):
    def __init__(self, llm, complexity="medium"):
        logger.info("❓ Initializing QuestionStrategy...")
        logger.info(f"🧠 LLM provided: {type(llm).__name__}")
        logger.info(f"🎯 Initial complexity: {complexity}")
        
        self.llm = llm
        self.complexity = complexity
        
        logger.info("🔧 Setting up prompt template...")
        self._set_prompt()
        
        logger.info("✅ QuestionStrategy initialized successfully")
    
    def _set_prompt(self):
        """Set up the prompt template based on complexity level."""
        logger.debug(f"📋 Setting prompt for complexity: {self.complexity}")
        
        complexity_instructions = {
            "easy": "Generate simple, basic questions that test understanding of key facts and definitions.",
            "medium": "Generate moderately challenging questions that require analysis and understanding of concepts.",
            "hard": "Generate complex questions that require critical thinking, analysis, and synthesis of information."
        }
        
        instruction = complexity_instructions.get(self.complexity, complexity_instructions["medium"])
        logger.debug(f"📝 Selected instruction: {instruction}")
        
        try:
            self.prompt = ChatPromptTemplate.from_template(f"""
            You are a helpful assistant tasked with generating question-answer pairs for study purposes.

            Text:
            {{context}}

            {instruction}
            Generate {{Questions}} meaningful questions based only on the above text. 

            IMPORTANT: Format your output exactly as shown below with no additional text, explanations, or formatting:

            Q1: [question text]
            Q2: [question text]
            Q3: [question text]
            """)
            
            logger.debug("⛓️ Building QA chain...")
            self.qa_chain = self.prompt | self.llm | StrOutputParser()
            
            logger.debug("✅ Prompt and chain setup completed")
            
        except Exception as e:
            logger.error(f"❌ Error setting up prompt: {str(e)}")
            raise

    def set_complexity(self, complexity):
        """Change complexity level with synonym mapping and fuzzy matching."""
        import difflib
        
        logger.info(f"🔄 Changing complexity from '{self.complexity}' to '{complexity}'")
        
        original_complexity = complexity
        complexity = complexity.lower().strip()
        
        # Handle synonyms first
        logger.debug("🔍 Checking synonyms...")
        synonyms = {
            "challenging": "hard", "difficult": "hard", "tough": "hard",
            "simple": "easy", "basic": "easy", "beginner": "easy", 
            "moderate": "medium", "average": "medium", "normal": "medium"
        }
        
        if complexity in synonyms:
            mapped_complexity = synonyms[complexity]
            logger.info(f"✅ Synonym '{complexity}' mapped to '{mapped_complexity}'")
            self.complexity = mapped_complexity
            self._set_prompt()
            return
        
        # Check exact match
        logger.debug("🎯 Checking exact matches...")
        valid_options = ["easy", "medium", "hard"]
        if complexity in valid_options:
            logger.info(f"✅ Exact match found: '{complexity}'")
            self.complexity = complexity
            self._set_prompt()
            return
        
        # Fuzzy matching against synonyms first
        logger.debug("🔍 Performing fuzzy matching...")
        all_options = list(synonyms.keys()) + valid_options
        matches = difflib.get_close_matches(complexity, all_options, n=1, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            similarity = difflib.SequenceMatcher(None, complexity, best_match).ratio()
            logger.info(f"🎯 Fuzzy match: '{complexity}' → '{best_match}' ({similarity:.0%} confidence)")
            print(f"'{complexity}' matched to '{best_match}' ({similarity:.0%} confidence)")
            
            # Map to final complexity
            final_complexity = synonyms.get(best_match, best_match)
            self.complexity = final_complexity
            logger.info(f"✅ Final complexity set to: '{final_complexity}'")
            self._set_prompt()
        else:
            logger.error(f"❌ No valid complexity match found for: '{original_complexity}'")
            raise ValueError("Please use: 'easy', 'medium', 'hard', or synonyms like 'challenging', 'simple'")

    def parse_qa_pairs(self, qa_output):
        """Parse the QA output into structured question-answer pairs."""
        logger.debug("📝 Parsing QA pairs from output...")
        
        try:
            qa_pairs = []
            lines = qa_output.strip().split('\n')
            total_lines = len(lines)
            
            logger.debug(f"📄 Processing {total_lines} lines...")
            
            i = 0
            parsed_count = 0
            
            # Process lines with progress tracking
            with tqdm(total=total_lines, desc="📝 Parsing QA pairs", unit="line", leave=False) as pbar:
                while i < len(lines):
                    q_match = re.match(r'Q(\d+):\s*(.+)', lines[i])
                    if q_match and i + 1 < len(lines):
                        question = q_match.group(2).strip()
                        # Use raw string for regex and escape the number for safety
                        a_match = re.match(rf'A{q_match.group(1)}:\s*(.+)', lines[i + 1])
                        if a_match:
                            answer = a_match.group(1).strip()
                            qa_pairs.append({'question': question, 'answer': answer})
                            parsed_count += 1
                            logger.debug(f"✅ Parsed Q{q_match.group(1)}: {question[:50]}...")
                            i += 2
                            pbar.update(2)
                        else:
                            i += 1
                            pbar.update(1)
                    else:
                        i += 1
                        pbar.update(1)
            
            logger.info(f"✅ QA parsing completed - {parsed_count} pairs extracted from {total_lines} lines")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"❌ Error parsing QA pairs: {str(e)}")
            raise

    def validate_input(self, doc):
        """Validate that the document is a Document instance with content."""
        logger.debug("🔍 Validating input document...")
        
        is_valid = (isinstance(doc, Document) and 
                   hasattr(doc, 'page_content') and 
                   len(doc.page_content.strip()) > 0)
        
        if is_valid:
            content_length = len(doc.page_content)
            logger.debug(f"✅ Document validation passed - Content length: {content_length} characters")
        else:
            logger.warning("❌ Document validation failed")
            if not isinstance(doc, Document):
                logger.warning(f"   - Not a Document instance: {type(doc)}")
            elif not hasattr(doc, 'page_content'):
                logger.warning("   - Missing page_content attribute")
            else:
                logger.warning("   - Empty or whitespace-only content")
        
        return is_valid
    
    def run(self, doc, questions, complexity='simple'):
        """Generate questions from the given document."""
        logger.info(f"🚀 Starting question generation...")
        logger.info(f"❓ Number of questions to generate: {questions}")
        logger.info(f"🎯 Complexity level: {complexity}")
        
        try:
            # Validate input
            if not self.validate_input(doc):
                logger.error("❌ Input validation failed")
                raise ValueError("Input must be a Document with non-empty page_content")
            
            # Update complexity if provided
            if complexity is not None and complexity != self.complexity:
                logger.info(f"🔄 Updating complexity to: {complexity}")
                self.set_complexity(complexity)
            
            # Generate QA output
            logger.info("🤖 Invoking QA generation chain...")
            with tqdm(total=1, desc="🔄 Generating questions", unit="task") as pbar:
                qa_output = self.qa_chain.invoke({
                    "context": doc.page_content,
                    "Questions": questions
                })
                pbar.update(1)
            
            logger.info("✅ QA generation completed")
            logger.info(f"📄 Generated output length: {len(qa_output)} characters")
            
            # Parse the output
            logger.info("📝 Parsing generated QA pairs...")
            parsed_qa = self.parse_qa_pairs(qa_output)
            
            # Log results
            logger.info(f"✅ Question generation successful!")
            logger.info(f"📊 Generated {len(parsed_qa)} question-answer pairs")
            
            # Print outputs (preserve original functionality)
            print(qa_output)
            print(parsed_qa)

            # Prepare result
            result = {
                "pdf_id": doc.metadata.get("pdf_id"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "qa_output": qa_output,
                "parsed_qa": parsed_qa
            }
            
            logger.info(f"📋 Result prepared for pdf_id: {result['pdf_id']}, chunk_id: {result['chunk_id']}")
            return result

        except Exception as e:
            error_msg = f"QA generation failed for Document {doc.metadata}: {e}"
            logger.error(f"❌ {error_msg}")
            print(f"❌ {error_msg}")
            
            # Provide helpful suggestions
            if "complexity" in str(e).lower():
                logger.error("💡 Suggestion: Check complexity parameter - use 'easy', 'medium', or 'hard'")
            elif "context" in str(e).lower():
                logger.error("💡 Suggestion: Check document content - ensure it's not empty")
            elif "model" in str(e).lower():
                logger.error("💡 Suggestion: Check LLM model availability and configuration")
            
            return None