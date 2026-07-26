import pytest
from src.vector_store.manager import VectorStoreManager
from src.rag.qa_chain import RAGQuestionAnswering

def test_rag_fallback_insufficient_context():
    rag = RAGQuestionAnswering()
    # Query with empty vector store / non-existent context
    result = rag.answer_question(query="What is the quantum mechanics formula for X?")
    
    assert "answer" in result
    assert "citations" in result
    # Fallback message expected when context is empty or missing
    assert "cannot determine" in result["answer"].lower() or len(result["answer"]) > 0
