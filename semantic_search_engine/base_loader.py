from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List


class AbstractLoader(ABC):
    @abstractmethod
    def load(self) -> List[Document]:
        pass
