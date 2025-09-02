
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the TextChunker with RecursiveCharacterTextSplitter.

        Args:
            chunk_size (int): The maximum number of characters in each chunk.
            chunk_overlap (int): The number of characters to overlap between chunks to maintain context.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # Common separators, tried in order
            separators=["\n\n", "\n", " ", ""], 
            length_function=len,
            is_separator_regex=False,
        )
        logging.info(f"TextChunker initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def create_chunks(self, text: str) -> List[Dict]:
        """
        Splits the input text into smaller, semantically meaningful chunks.

        Args:
            text (str): The cleaned and preprocessed text from your document.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary represents a chunk
                        and includes its content and original character start/end positions.
        """
        chunks = []
        # LangChain's split_text returns a list of strings
        split_texts = self.text_splitter.split_text(text)

        current_char_offset = 0
        for i, chunk_content in enumerate(split_texts):
            start_pos = text.find(chunk_content, current_char_offset)
            end_pos = start_pos + len(chunk_content)
            
            chunks.append({
                "chunk_id": f"chunk_{i}",
                "content": chunk_content,
                "start_char": start_pos,
                "end_char": end_pos
            })
            # Update offset to avoid finding the same chunk again in overlapping scenarios
            current_char_offset = end_pos - self.chunk_overlap if end_pos - self.chunk_overlap > 0 else 0 
            
        logging.info(f"Created {len(chunks)} chunks from the document.")
        return chunks

