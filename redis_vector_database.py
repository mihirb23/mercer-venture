

import logging
from typing import List, Dict
import numpy as np

from redis import Redis
from redisvl.index import SearchIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RedisVectorDBManager:
    def __init__(self, redis_url: str, index_name: str, vector_dimension: int):
        """
        Initializes the Redis client and sets up the search index.
        """
        try:
            # Connect to Redis directly
            self.redis_client = Redis.from_url(redis_url, decode_responses=False)
            
            # Define schema as dictionary (compatible with RedisVL > 0.2.0)
            schema = {
                "index": {"name": index_name, "prefix": f"doc:{index_name}"},
                "fields": [
                    {"name": "vector", "type": "vector", "attrs": {"dims": vector_dimension, "algorithm": "flat", "distance_metric": "cosine"}},
                    {"name": "content", "type": "text"},
                    {"name": "source_doc", "type": "tag"},
                    {"name": "chunk_id", "type": "tag"},
                    {"name": "start_char", "type": "text"},
                    {"name": "end_char", "type": "text"}
                ]
            }
            # Create the actual SearchIndex object
            self.search_index = SearchIndex.from_dict(schema)
            self.search_index.set_client(self.redis_client)

            if not self.search_index.exists():
                self.search_index.create(overwrite=True)
                logging.info(f"RediSearch index '{index_name}' created.")
            else:
                logging.info(f"Connecting to existing RediSearch index '{index_name}'.")

            logging.info(f"Successfully connected to Redis at {redis_url}, index: {index_name}")

        except Exception as e:
            logging.error(f"Failed to initialize Redis or RediSearch index: {e}", exc_info=True)
            raise

    def upsert_vectors(self, vectors_data: List[Dict]):
        if not vectors_data:
            logging.warning("No vectors to upsert.")
            return
        try:
            docs_to_upsert = []
            for item in vectors_data:
            # Convert vector list to numpy float32 array, then to bytes
                vector_bytes = np.array(item["vector"], dtype=np.float32).tobytes()
            
                doc = {
                    "id": item["id"],
                    "vector": vector_bytes,      # <-- pass bytes, not list/ndarray
                    "content": item["content"],
                    "source_doc": item["source_doc"],
                    "chunk_id": item["chunk_id"],
                    "start_char": item["start_char"],
                    "end_char": item["end_char"],
                }
                docs_to_upsert.append(doc)
            self.search_index.load(docs_to_upsert)
            logging.info(f"Successfully upserted {len(vectors_data)} vectors into Redis index '{self.search_index.name}'.")
        except Exception as e:
            logging.error(f"Failed to upsert vectors to Redis: {e}", exc_info=True)
            raise
