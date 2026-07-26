import os
import pytest
import fitz  # PyMuPDF
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker

@pytest.fixture
def sample_pdf_path(tmp_path):
    """Creates a temporary sample PDF document for testing."""
    pdf_file = tmp_path / "sample_paper.pdf"
    doc = fitz.open()
    
    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Artificial Intelligence Research Paper.\nThis document introduces transformer architectures for deep learning models.")
    
    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Methodology and Performance Evaluation.\nWe evaluate precision, recall, and vector similarity metrics across dataset benchmarks.")

    doc.save(str(pdf_file))
    doc.close()
    return str(pdf_file)

def test_pdf_parser_extraction(sample_pdf_path):
    parser = PDFParser()
    extracted = parser.extract_text_with_metadata(sample_pdf_path, "doc_test_123")
    
    assert len(extracted) == 2
    assert extracted[0]["page_number"] == 1
    assert extracted[1]["page_number"] == 2
    assert "Artificial Intelligence" in extracted[0]["text"]
    assert "Methodology" in extracted[1]["text"]

def test_document_chunker(sample_pdf_path):
    parser = PDFParser()
    extracted = parser.extract_text_with_metadata(sample_pdf_path, "doc_test_123")
    
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.create_chunks(extracted)
    
    assert len(chunks) >= 2
    assert all("chunk_id" in c for c in chunks)
    assert all("page_number" in c for c in chunks)
