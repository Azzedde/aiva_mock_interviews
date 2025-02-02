from PyPDF2 import PdfReader
import re

class PDFProcessor:
    """
    Utility class for PDF text extraction and processing.
    """
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and format text from PDF.
        """
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        text = re.sub(r',(?=[^\s])', ', ', text)
        text = re.sub(r'(?<=[.!?])\s{2,}', '\n\n', text)
        text = text.replace('•', '\n• ')
        text = re.sub(r'([a-z])([A-Z])', r'\1\n\2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extract and clean text from a PDF file.
        """
        try:
            reader = PdfReader(pdf_path)
            text_parts = []

            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    cleaned_text = PDFProcessor.clean_text(page_text)
                    text_parts.append(cleaned_text)

            full_text = '\n\n'.join(text_parts)
            return PDFProcessor.clean_text(full_text)

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
