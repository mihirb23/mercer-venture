
import os
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pandas as pd
import logging
from typing import List, Tuple

# --- Azure Integration ---
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    Handles text and table extraction, using Azure AI Document Intelligence for scanned PDFs.
    """
    def __init__(self):
        try:
            self.azure_endpoint = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
            self.azure_key = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
            self.document_intelligence_client = DocumentIntelligenceClient(
                endpoint=self.azure_endpoint, credential=AzureKeyCredential(self.azure_key)
            )
        except KeyError:
            logging.error("Azure credentials not found. Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY.")
            self.document_intelligence_client = None

    def process_document(self, file_path: str) -> Tuple[str, List[pd.DataFrame]]:
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Currently, only PDF files are supported.")
        
        logging.info(f"Processing document: {file_path}")
        
        if self._is_scanned_pdf(file_path):
            logging.info("Detected scanned PDF. Using Azure AI Document Intelligence.")
            if not self.document_intelligence_client:
                raise ConnectionError("Azure client not initialized. Check your credentials.")
            return self._extract_with_azure(file_path)
        else:
            logging.info("Detected digital PDF. Using direct text extraction.")
            return self._extract_from_digital_pdf(file_path)

    def _is_scanned_pdf(self, file_path: str, char_threshold: int = 100) -> bool:
        try:
            with pdfplumber.open(file_path) as pdf:
                # Check the first few pages for text content to be more robust
                total_chars = 0
                for page in pdf.pages[:min(3, len(pdf.pages))]: # Check up to the first 3 pages
                    text = page.extract_text()
                    if text:
                        total_chars += len(text.strip())
                return total_chars < char_threshold * min(3, len(pdf.pages))
        except Exception:
            # If any error occurs opening with pdfplumber, assume it's scanned.
            return True

    def _extract_from_digital_pdf(self, file_path: str) -> Tuple[str, List[pd.DataFrame]]:
        full_text = ""
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                full_text += f"\n\n--- PAGE {i+1} ---\n\n" + page_text
        try:
            logging.info("Extracting tables with Camelot...")
            camelot_tables = camelot.read_pdf(file_path, flavor='lattice', pages='all')
            tables = [tbl.df for tbl in camelot_tables if tbl.accuracy > 80]
            logging.info(f"Found {len(tables)} high-quality tables with Camelot.")
        except Exception as e:
            logging.warning(f"Camelot table extraction failed: {e}")
        return full_text.strip(), tables

    def _extract_with_azure(self, file_path: str) -> Tuple[str, List[pd.DataFrame]]:
        """
        Uses Azure's "prebuilt-read" model to perform OCR on a document.
        """
        with open(file_path, "rb") as f:
            # --- THIS IS THE CORRECTED PART ---
            # The argument for the document content is 'body'
            poller = self.document_intelligence_client.begin_analyze_document(
                model_id="prebuilt-read", body=f, content_type="application/octet-stream"
            )
        
        result = poller.result()
        full_text = result.content
        
        azure_tables = []
        if result.tables:
            logging.info(f"Azure extracted {len(result.tables)} tables.")
            for table in result.tables:
                df_rows = []
                for row_idx in range(table.row_count):
                    row_cells = [None] * table.column_count
                    for cell in table.cells:
                        if cell.row_index == row_idx:
                            row_cells[cell.column_index] = cell.content
                    df_rows.append(row_cells)
                
                if df_rows and len(df_rows) > 1:
                    df = pd.DataFrame(df_rows[1:], columns=df_rows[0])
                    azure_tables.append(df)
        
        return full_text, azure_tables
