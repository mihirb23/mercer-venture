import logging
import pandas as pd
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    Extracts text and tables from PDFs using docTR on every PDF (scanned or digital).
    Outputs:
      - fields: extracted key-value pairs (incl. from table rows)
      - tables: all found tables as list of lists, per page
      - ocr_text: plain extracted text, all pages
    """
    def __init__(self):
        self.doctr_model = ocr_predictor(pretrained=True)

    def process_document(self, file_path):
        logging.info(f"Processing document: {file_path} with docTR")
        doc = DocumentFile.from_pdf(file_path)
        result = self.doctr_model(doc)

        fields = {}
        tables = []
        all_lines = []

        # Step 1: Collect lines for field and table detection
        for page_idx, page in enumerate(result.pages, 1):
            page_lines = []
            for block in page.blocks:
                for line in block.lines:
                    words = [word.value for word in line.words]
                    line_text = " ".join(words).strip()
                    if line_text:
                        all_lines.append(line_text)
                        page_lines.append(words)

            # Table detection: contiguous blocks with >2 columns and numbers
            detected_tables = []
            current_table = []
            for row in page_lines:
                num_columns = len(row)
                num_with_digits = sum(any(char.isdigit() for char in cell) for cell in row)
                if num_columns > 2 and num_with_digits > 1:
                    current_table.append(row)
                else:
                    if len(current_table) > 1:
                        detected_tables.append(current_table)
                    current_table = []
            if len(current_table) > 1:
                detected_tables.append(current_table)

            # Save tables as DataFrames
            for t in detected_tables:
                df = pd.DataFrame(t)
                tables.append({"page": page_idx, "table": df})

        # Step 2: Key-value extraction from all lines
        for line in all_lines:
            # Key-value (colon)
            if ':' in line:
                key, value = line.split(':', 1)
                fields[key.strip()] = value.strip()
            # Multi-col layout: double-space or tab split (fallback)
            else:
                parts = [p.strip() for p in line.split('  ') if p.strip()]
                if len(parts) == 2:
                    fields[parts[0]] = parts[1]

        # Step 3: Smart nesting of fields for table rows
        for table_entry in tables:
            df = table_entry['table']
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue
            headers = list(df.iloc[0])
            for i in range(1, df.shape[0]):
                first_cell = str(df.iloc[i, 0]).strip()
                if first_cell:
                    row_dict = {headers[j]: df.iloc[i, j]
                                for j in range(1, df.shape[1]) if j < len(headers)}
                    if any(v for v in row_dict.values()):
                        fields[first_cell] = row_dict

        ocr_text = "\n".join(all_lines)
        return {
            "fields": fields,
            "tables": [
                {"page": t["page"], "table": t["table"].values.tolist()} for t in tables
            ],
            "ocr_text": ocr_text
        }
