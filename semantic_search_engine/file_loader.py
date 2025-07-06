from .base_loader import AbstractLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_unstructured import UnstructuredLoader
from .logger import get_logger


logger = get_logger(__name__)


class FileLoader(AbstractLoader):
    def __init__(self, path: str):
        self.path = path

    def load(self):
        logger.info(f"Loading file {self.path}")
        if self.path.lower().endswith(".pdf"):
            loader = PyMuPDF4LLMLoader(
                file_path=self.path,
                mode="page",
                extract_images=False
            )
        else:
            loader = UnstructuredLoader(
                file_path=self.path,
                chunking_strategy="basic",
                max_characters=1_000_000,
                include_orig_elements=False,
            )
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} elements from file")
        return docs