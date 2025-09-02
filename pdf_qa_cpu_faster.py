# pdf_qa_cpu_faster.py

import os
import io
import json
import fitz
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

from transformers import AutoProcessor, AutoModelForVision2Seq
from colpali_engine.models import ColQwen2, ColQwen2Processor
import faiss

# 1. Document Processor
class DocumentProcessor:
    def pdf_to_images(self, pdf_path, dpi=150):
        images = []
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)
        pdf_document.close()
        return images

# 2. Smaller Qwen2-VL Model for Q&A (2B parameters)
class VLM_QwenExtractor:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cpu"
        print(f"Loading {model_name} on CPU...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map={"": "cpu"},
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print("Qwen2-VL-2B loaded successfully.")

    def answer_question(self, images, question):
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the following images."},
            {
                "role": "user",
                "content": [{"type": "text", "text": question}] +
                           [{"type": "image", "image": img} for img in images[:3]]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt],
            images=images[:3],
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(**inputs, max_new_tokens=512, temperature=0.2)

        return self.processor.batch_decode(output_tokens, skip_special_tokens=True)[0]

# 3. ColQwen2 Embeddings (same as before)
class ColQwenEmbedder:
    def __init__(self):
        self.device = "cpu"
        print("Loading ColQwen2 on CPU...")
        self.model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0",
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        print("ColQwen2 loaded successfully.")

    def image_embeddings(self, images):
        embs = []
        with torch.no_grad():
            for img in images:
                inputs = self.processor.process_images([img]).to(self.device)
                emb = self.model(**inputs)
                embs.append(emb.cpu().numpy())
        return np.vstack(embs)

    def text_embedding(self, text):
        with torch.no_grad():
            inputs = self.processor.process_queries([text]).to(self.device)
            return self.model(**inputs).cpu().numpy()

# 4. FAISS Vector Store
class VectorStore:
    def __init__(self, dim=1024):
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, embeddings, pages):
        self.index.add(embeddings.astype("float32"))
        self.docs.extend(pages)

    def search(self, query_emb, k=3):
        distances, idxs = self.index.search(query_emb.astype("float32"), k)
        results = []
        for i, dist in zip(idxs[0], distances[0]):
            results.append({"page": self.docs[i], "score": float(dist)})
        return results

# 5. Main System
class PDF_QA_System:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vlm = VLM_QwenExtractor()
        self.embedder = ColQwenEmbedder()
        self.vector_store = VectorStore()
        self.images = []

    def process_pdf(self, pdf_path):
        print(f"Processing PDF: {pdf_path}")
        self.images = self.doc_processor.pdf_to_images(pdf_path)
        print(f"- Extracted {len(self.images)} pages as images.")
        embeddings = self.embedder.image_embeddings(self.images)
        self.vector_store.add(embeddings, list(range(len(self.images))))
        print("- Embeddings generated and stored.")

    def query(self, question):
        print(f"Asking: {question}")
        q_emb = self.embedder.text_embedding(question)
        results = self.vector_store.search(q_emb, k=2)
        context_imgs = [self.images[r['page']] for r in results]
        return self.vlm.answer_question(context_imgs, question)

# 6. Run as Script
if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Change to your PDF filename here
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    system = PDF_QA_System()
    system.process_pdf(pdf_path)

    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() in ["exit", "quit"]:
            break
        answer = system.query(question)
        print(f"\nAnswer:\n{answer}\n")
