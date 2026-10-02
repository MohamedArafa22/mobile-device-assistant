import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode_documents( self, documents: list[str])-> np.ndarray:

        return self.model.encode(documents,normalize_embeddings=True)

    def encode_query(self, query: str)-> np.ndarray:

        return self.model.encode(query,normalize_embeddings=True)