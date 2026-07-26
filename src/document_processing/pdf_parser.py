import os
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional

class PDFParser:
    """PDF text parser preserving page structure and page metadata."""

    def extract_text_with_metadata(self, pdf_path: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Extract text page by page from a PDF file.
        Returns a list of dicts containing doc_id, page_number, and text.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        extracted_pages = []
        doc = fitz.open(pdf_path)
        
        try:
            total_pages = len(doc)
            for page_idx in range(total_pages):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                if text:
                    extracted_pages.append({
                        "doc_id": doc_id,
                        "file_name": os.path.basename(pdf_path),
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "text": text
                    })
        finally:
            doc.close()

        return extracted_pages

    def get_document_summary_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract high-level document metadata like title, author, total pages."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            meta = doc.metadata or {}
            total_pages = len(doc)
            return {
                "file_name": os.path.basename(pdf_path),
                "total_pages": total_pages,
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "keywords": meta.get("keywords", "")
            }
        finally:
            doc.close()
