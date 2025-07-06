import os

import numpy as np

from .embedder import SentenceTransformerEmbedder
from .file_loader import FileLoader
from .logger import get_logger
from pathlib import Path
from .settings import Settings
from .splitter import TextSplitter
from typing import List, Dict
from .url_loader import URLLoader
from .vectorstore import FaissIndex

if Settings.USE_QDRANT:
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct


logger = get_logger(__name__)


class SearchEngine:
    def __init__(self):
        self.splitter = TextSplitter()
        self.embedder = SentenceTransformerEmbedder()
        dim = getattr(self.embedder.model.config, "hidden_size", None)
        if dim is None:
            sample_vec = self.embedder.embed_query("test")
            dim = len(sample_vec)

        self.use_qdrant = Settings.USE_QDRANT
        if self.use_qdrant:
            self.qdrant = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333")
            )
            self.collection = Settings.QDRANT_COLLECTION

        self.faiss = FaissIndex(dim)
        data_dir = Path(Settings.DATA_DIR)
        sources = [
            str(p) for p in data_dir.glob("*")
            if p.suffix.lower() in (".pdf", ".txt")
        ]
        if sources:
            logger.info(f"Startup ingestion of {len(sources)} sources...")
            self.ingest(sources)


    def ingest(self, sources: List[str]):
        docs = []
        for src in sources:
            loader = URLLoader(src) if src.startswith("http") else FileLoader(src)
            docs.extend(loader.load())

        chunks = self.splitter.split(docs)
        texts = [c.page_content for c in chunks]
        vectors = self.embedder.embed_batch(texts)
        metas = [{"text": t, "meta": d.metadata} for t, d in zip(texts, chunks)]

        if self.use_qdrant:
            points = [
                PointStruct(id=i, vector=v.tolist(), payload=m)
                for i, (v, m) in enumerate(zip(vectors, metas))
            ]
            self.qdrant.upload_collection(
                collection_name=self.collection,
                points=points,
                parallel=4,
            )
            logger.info(f"Ingested {len(points)} items into Qdrant '{self.collection}'")
        else:
            self.faiss.add(np.array(vectors), metas)
            logger.info(f"Added {len(vectors)} vectors to FAISS index")


    def search(self, query: str) -> List[Dict]:
        if not query:
            return []

        vec = self.embedder.embed_query(query)
        qvec = np.array(vec).reshape(1, -1)

        if self.use_qdrant:
            hits = self.qdrant.search(
                collection_name=self.collection,
                query_vector=qvec[0].tolist(),
                limit=Settings.TOP_K,
            )
            return [{"score": h.score, **h.payload} for h in hits]

        k0 = Settings.TOP_K * 5
        distances, idxs = self.faiss.index.search(qvec.astype("float32"), k0)
        unique, seen = [], set()
        for idx in idxs[0]:
            if idx < 0:
                continue
            meta = self.faiss.metadata[idx]
            
            src = meta.get("meta", {}).get("source") or meta.get("source")
            if src in seen:
                continue
            seen.add(src)
            unique.append(meta)
            if len(unique) >= Settings.TOP_K:
                break
        return unique