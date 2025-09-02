import os
import logging
import json
import numpy as np
import torch

from processing_the_documents1 import DocumentProcessor
from text_preprocessing1 import TextPreprocessor
from text_chunking import TextChunker
from embedding_generator import EmbeddingGenerator
from redis_vector_database import RedisVectorDBManager
from bilstm_bigru_model import BiLSTMInference, BiGRUInference
from tokenizer import simple_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def full_pipeline_with_rag_setup(pdf_path: str, bilstm_model=None, bigru_model=None):
    logging.info(f"Starting full pipeline for: {pdf_path}")

    # Instantiate DocumentProcessor and get OCR result dict
    doc_processor = DocumentProcessor()
    result = doc_processor.process_document(pdf_path)
    fields = result["fields"]
    tables = result["tables"]
    raw_text = result["ocr_text"]
    logging.info("Document OCR extraction complete.")

    # Preprocess text
    text_preprocessor = TextPreprocessor()
    processed_data = text_preprocessor.process(raw_text)
    cleaned_text = processed_data["cleaned_text"]
    logging.info("Text preprocessing complete.")

    # Add extracted fields and tables to processed data JSON
    processed_data["extracted_fields"] = fields
    processed_data["extracted_tables"] = tables

    # Document-level classification
    if bilstm_model is not None:
        doc_pred_bilstm = bilstm_model.infer(cleaned_text)
        processed_data["bilstm_doc_classification"] = doc_pred_bilstm
        logging.info(f"BiLSTM classification: {doc_pred_bilstm}")

    if bigru_model is not None:
        doc_pred_bigru = bigru_model.infer(cleaned_text)
        processed_data["bigru_doc_classification"] = doc_pred_bigru
        logging.info(f"BiGRU classification: {doc_pred_bigru}")

    # Save outputs
    output_dir = "processed_documents"
    os.makedirs(output_dir, exist_ok=True)
    document_name = os.path.splitext(os.path.basename(pdf_path))[0]

    with open(os.path.join(output_dir, f"{document_name}_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    with open(os.path.join(output_dir, f"{document_name}_processed_data.json"), "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved cleaned text and processed data for {document_name}.")

    # Chunking cleaned text
    text_chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    chunks = text_chunker.create_chunks(cleaned_text)
    logging.info(f"Created {len(chunks)} chunks.")

    # Chunk-level classification
    if bilstm_model is not None:
        for chunk in chunks:
            chunk["bilstm_classification"] = bilstm_model.infer(chunk["content"])
    if bigru_model is not None:
        for chunk in chunks:
            chunk["bigru_classification"] = bigru_model.infer(chunk["content"])

    # Generate embeddings for chunks
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    chunk_texts = [chunk["content"] for chunk in chunks]
    chunk_embeddings = embedding_generator.generate_embeddings(chunk_texts)
    logging.info("Embeddings generated for all chunks.")

    # Upsert to Redis vector DB
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_index_name = "insurance_docs_index"
    embedding_dim = 384
    redis_db_manager = RedisVectorDBManager(redis_url, redis_index_name, embedding_dim)

    docs_to_upsert = []
    for i, chunk in enumerate(chunks):
        doc = {
            "id": f"{document_name}_chunk_{i}",
            "vector": np.array(chunk_embeddings[i], dtype=np.float32).tobytes(),
            "content": chunk["content"],
            "chunk_id": f"{document_name}_chunk_{i}",
            "start_char": str(chunk["start_char"]),
            "end_char": str(chunk["end_char"]),
            "source_doc": os.path.basename(pdf_path),
        }
        if "bilstm_classification" in chunk:
            doc["bilstm_classification"] = json.dumps(chunk["bilstm_classification"])
        if "bigru_classification" in chunk:
            doc["bigru_classification"] = json.dumps(chunk["bigru_classification"])
        docs_to_upsert.append(doc)

    redis_db_manager.upsert_vectors(docs_to_upsert)
    logging.info(f"Chunks and embeddings stored for {document_name}.")

if __name__ == "__main__":
    embedding_matrix = np.load("embedding_matrix.npy")
    padding_idx = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bilstm_model = BiLSTMInference(
        model_path="your_bilstm_model.pth",
        embedding_matrix=embedding_matrix,
        pooling="mask_average",
        device=device,
        padding_idx=padding_idx
    )
    bilstm_model.set_tokenizer(simple_tokenizer)

    bigru_model = BiGRUInference(
        model_path="your_bigru_model.pth",
        embedding_matrix=embedding_matrix,
        pooling="mask_average",
        device=device,
        padding_idx=padding_idx
    )
    bigru_model.set_tokenizer(simple_tokenizer)

    documents_folder = "documents/"
    try:
        pdf_files = [f for f in os.listdir(documents_folder) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            file_path = os.path.join(documents_folder, pdf_file)
            print("\n" + "#" * 70)
            print(f"# PROCESSING FILE: {pdf_file}")
            print("#" * 70 + "\n")
            try:
                full_pipeline_with_rag_setup(file_path, bilstm_model=bilstm_model, bigru_model=bigru_model)
            except Exception as e:
                logging.error(f"Error processing {pdf_file}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Unexpected error in batch processing: {e}", exc_info=True)
