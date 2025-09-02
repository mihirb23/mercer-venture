
from sentence_transformers import SentenceTransformer
import torch
import logging
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the EmbeddingGenerator with a pre-trained SentenceTransformer model.

        Args:
            model_name (str): The name of the SentenceTransformer model to load.
                              "all-MiniLM-L6-v2" is a good balance of performance and size [6][16].
        """
        try:
            # Load the model, automatically using CUDA/MPS if available [6]
            self.model = SentenceTransformer(model_name)
            logging.info(f"Embedding model '{model_name}' loaded successfully.")
            logging.info(f"Model max sequence length: {self.model.max_seq_length}")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of text chunks.

        Args:
            texts (List[str]): A list of text strings (chunks) to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []
        
        # Encode texts to embeddings. normalize_embeddings=True makes vectors unit length,
        # which allows using dot-product for similarity instead of cosine similarity,
        # often leading to faster computations [16].
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        logging.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")
        return embeddings.tolist()

