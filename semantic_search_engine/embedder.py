import torch


from .logger import get_logger
from .settings import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from typing import List


logger = get_logger(__name__)


class SentenceTransformerEmbedder:
    def __init__(self):
        logger.info(f"Loading HuggingFace model '{Settings.EMBED_MODEL}' on CPU")
        self.tokenizer = AutoTokenizer.from_pretrained(Settings.EMBED_MODEL)
        self.model = AutoModel.from_pretrained(Settings.EMBED_MODEL)
        self.model.eval()
        logger.info("Model ready on CPU")


    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        for k, v in inputs.items():
            inputs[k] = v.to("cpu")
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state 
            
        mask = inputs["attention_mask"].unsqueeze(-1) 
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        embeddings = (summed / counts).cpu().tolist()
        return embeddings


    def embed_query(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]