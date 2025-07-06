import faiss


import numpy as np


from .logger import get_logger
from .settings import Settings
from typing import List, Dict, Any


logger = get_logger(__name__)


class FaissIndex:
    def __init__(self, dim: int):
        logger.info(f"Creating FAISS HNSW index (dim={dim})")
        idx = faiss.IndexHNSWFlat(dim, Settings.HNSW_M)
        idx.hnsw.efSearch = Settings.HNSW_EFSEARCH
        self.index = idx
        self.metadata: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metas: List[Dict[str, Any]]):
        n = vectors.shape[0]
        logger.info(f"Adding {n} vectors to FAISS index")
        self.index.add(vectors.astype("float32"))
        self.metadata.extend(metas)

    def save(self, path: str):
        faiss.write_index(
            self.index, path,
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        logger.info(f"FAISS index saved (mmap) at {path}")

    @classmethod
    def load(cls, path: str):
        idx = faiss.read_index(path, faiss.IO_FLAG_MMAP)
        inst = cls.__new__(cls)
        inst.index = idx
        inst.metadata = []
        return inst

    def query(self, qvec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        distances, idxs = self.index.search(qvec.astype('float32'), top_k)
        results = []
        for i in idxs[0]:
            if i < 0 or i >= len(self.metadata):
                continue
            results.append(self.metadata[i])
        return results