import logging
import spacy
import re
from collections import Counter
import editdistance
from typing import List, Dict, Set, Tuple


class TextPreprocessor:
    """
    Cleans and structures text using spaCy, handling OCR artifacts,
    OOV words, and segmenting the document. Also extracts key-value fields.
    """
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        # Custom vocabulary based on your documents to improve OOV handling
        self.custom_vocabulary = {
            "leipig", "g0003524", "grh", "pte", "sgid_ts1",
            "policyholder", "deductible", "coinsurance", "hospitalisation"
        }

    def process(self, text: str) -> Dict:
        """
        Runs the full preprocessing pipeline on the input text.
        """
        logging.info("Starting text preprocessing...")

        # 1. Clean the text to fix spacing and artifacts
        cleaned_text = self._clean_text(text)

        # 2. Process the cleaned text with spaCy
        doc = self.nlp(cleaned_text)

        # 3. Handle Out-of-Vocabulary (OOV) words
        oov_words, corrections = self._handle_oov(doc)

        # 4. Segment the document into logical sections
        sections = self._segment_document(doc)

        # 5. Extract key-value pairs from cleaned text
        kv_pairs = self._extract_key_value_pairs(cleaned_text)

        logging.info("Text preprocessing completed.")

        return {
            "cleaned_text": cleaned_text,
            "sentences": [sent.text for sent in doc.sents],
            "tokens": [{'text': token.text, 'pos': token.pos_} for token in doc],
            "pos_tags": [(token.text, token.pos_) for token in doc],
            "oov_words": oov_words,
            "oov_corrections": corrections,
            "sections": sections,
            "key_value_fields": kv_pairs,
        }


    def _clean_text(self, text: str) -> str:
        """
        Cleans text by normalizing whitespace and fixing common OCR/extraction errors.
        """
        # Fix oddly spaced words like "com puter gen erated"
        text = re.sub(r'\b(\w)\s(?=\w\b)', r'\1', text)

        # Normalize newlines and spaces
        text = re.sub(r'(\s*\n\s*)+', '\n', text)  # Multiple newlines to one
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to one

        # Fix key-value pairs separated by many spaces, e.g., "Policyholder      LEIPIG"
        text = re.sub(r':\s+', ': ', text)

        # Remove page markers
        text = re.sub(r'--- PAGE \d+ ---', '', text)

        return text.strip()


    def _handle_oov(self, doc: spacy.tokens.Doc) -> Tuple[List[str], Dict[str, str]]:
        """
        Identifies OOV words and suggests corrections using edit distance.
        """
        oov_words = []
        corrections = {}

        known_words = set(self.nlp.vocab.strings) | self.custom_vocabulary

        for token in doc:
            if token.is_alpha and token.text.lower() not in known_words:
                oov_words.append(token.text)
                closest_word = min(
                    self.custom_vocabulary,
                    key=lambda word: editdistance.eval(token.text.lower(), word),
                    default=None
                )
                if closest_word and editdistance.eval(token.text.lower(), closest_word) <= 2:
                    corrections[token.text] = closest_word

        return list(set(oov_words)), corrections


    def _segment_document(self, doc: spacy.tokens.Doc) -> List[Dict]:
        """
        Splits the document into logical sections based on headers and keywords.
        """
        sections = []
        section_patterns = {
            "HEADER": r"^(GROUP INSURANCE POLICY|SCHEDULE OF BENEFITS|THE BC LIFE ASSURANCE COMPANY)",
            "POLICY_DETAILS": r"^(Subject|Policyholder|Policy No\.|Group Policy Number)",
            "BENEFITS_TABLE": r"^(SCHEDULE OF BENEFITS|PLAN 1|PLAN 2|PLAN 3|PLAN 4)",
            "LEGAL": r"^(Notwithstanding anything contained|Except to the extent expressly|This policy is protected)"
        }

        current_section = "UNKNOWN"
        current_content = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            matched_section = None
            for section_name, pattern in section_patterns.items():
                if re.search(pattern, sent_text, re.IGNORECASE):
                    matched_section = section_name
                    break

            if matched_section and matched_section != current_section:
                if current_content:
                    sections.append({"title": current_section, "content": "\n".join(current_content)})
                current_section = matched_section
                current_content = [sent_text]
            else:
                current_content.append(sent_text)

        if current_content:
            sections.append({"title": current_section, "content": "\n".join(current_content)})

        return sections

    def _extract_key_value_pairs(self, text: str) -> Dict[str, str]:
        """
        Extracts key-value pairs from text lines in the form 'Key: Value'.
        """
        kv_pairs = {}
        for line in text.splitlines():
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                value = parts[1].strip()
                if key and value:
                    kv_pairs[key] = value
        return kv_pairs
