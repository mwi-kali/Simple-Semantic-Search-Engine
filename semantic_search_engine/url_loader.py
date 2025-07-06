
from .base_loader import AbstractLoader
from langchain_community.document_loaders import WebBaseLoader
from .logger import get_logger


logger = get_logger(__name__)


class URLLoader(AbstractLoader):
    def __init__(self, url: str):
        self.url = url

    def load(self):
        logger.info(f"Fetching URL ... {self.url}")
        loader = WebBaseLoader(self.url)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document elements from web")
        return docs
