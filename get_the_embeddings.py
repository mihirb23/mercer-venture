import logging
from typing import List, Dict, Tuple

from redisvl.query import VectorQuery
from redisvl.index import SearchIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    def __init__(self, search_index: SearchIndex, embedding_generator):
        """
        Initializes the DocumentRetriever with a RediSearch index and an EmbeddingGenerator.

        Args:
            search_index (SearchIndex): The initialized RedisVL SearchIndex object.
            embedding_generator: An instance of your EmbeddingGenerator.
        """
        self.search_index = search_index
        self.embedding_generator = embedding_generator
        logging.info("DocumentRetriever initialized for Redis.")

    def retrieve_relevant_chunks(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Embeds the user query and retrieves the most relevant document chunks
        from the Redis vector database.

        Args:
            query_text (str): The user's question.
            top_k (int): The number of top relevant chunks to retrieve.

        Returns:
            List[Dict]: A list of dictionaries, each containing the content and metadata
                        of a retrieved relevant chunk.
        """
        if not query_text:
            logging.warning("Query text is empty. Returning no chunks.")
            return []

        try:
            # 1. Embed the user query
            query_embedding = self.embedding_generator.generate_embeddings([query_text])[0]
            logging.info(f"Generated embedding for query: '{query_text[:50]}...'")

            # 2. Perform similarity search in Redis
            # Create a vector query
            vector_query = VectorQuery(
                vector=query_embedding,
                num_results=top_k,
                return_fields=["content", "start_char", "end_char", "source_doc", "chunk_id", "__vector_score"],
                return_documents=True # Get the full document (hash) fields
            )

            # Perform the search
            query_results = self.search_index.query(vector_query)
            logging.info(f"Retrieved {len(query_results)} relevant chunks from Redis.")

            relevant_chunks = []
            for match in query_results:
                # RedisVL returns a dict with all fields, including the score from __vector_score
                relevant_chunks.append({
                    "chunk_id": match.get('chunk_id'),
                    "content": match.get('content'),
                    "score": match.get('__vector_score'),
                    "start_char": match.get('start_char'),
                    "end_char": match.get('end_char'),
                    "source_doc": match.get('source_doc')
                })
            
            return relevant_chunks

        except Exception as e:
            logging.error(f"Failed to retrieve relevant chunks from Redis: {e}")
            raise

