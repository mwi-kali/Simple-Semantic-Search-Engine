from pathlib import Path


class Settings:
    
    CHUNK_OVERLAP: int = 200   
    CHUNK_SIZE: int = 1000         
    DATA_DIR: Path = Path(__file__).parent.parent /  "data"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    HNSW_EFSEARCH: int = 32     
    HNSW_M: int = 16 
    LOG_LEVEL: str = "INFO"       
    PROMETHEUS_PORT: int = 8000   
    QDRANT_COLLECTION: str = "semantic_search"               
    TOP_K: int = 5                  
    USE_QDRANT: bool = False
    