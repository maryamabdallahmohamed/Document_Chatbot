import src.processors.base_preprocessor as BasePreprocessor 
import json
from langchain.schema import Document
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class JSONPreprocessor(BasePreprocessor.BasePreprocessor):
    def load_and_preprocess_data(self, file_path):
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        clean_texts = [self.clean_text(entry) for entry in raw_data if isinstance(entry, str)]
        return "\n".join(clean_texts)
    def process_documents_from_files(self, file_paths):
        documents = []

        for i, file_path in enumerate(file_paths):
            text = self.load_and_preprocess_data(file_path).strip()
            documents.append(
                Document(page_content=text, metadata={"pdf_id": i})
            )

        return documents
