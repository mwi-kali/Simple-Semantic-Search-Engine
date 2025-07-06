from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .settings import Settings
from typing import List


class TextSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
            add_start_index=True
        )

    def split(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)
