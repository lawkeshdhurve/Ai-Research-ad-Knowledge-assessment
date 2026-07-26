# Lazy Singleton Service Registry for Memory Efficiency
from typing import Optional

_vector_store = None
_predictor = None
_summarizer = None
_comparator = None
_qa_chain = None
_analytics = None

def get_vector_store():
    """Lazily initializes and returns singleton VectorStoreManager instance."""
    global _vector_store
    if _vector_store is None:
        from src.vector_store.manager import VectorStoreManager
        _vector_store = VectorStoreManager()
    return _vector_store

def get_predictor():
    """Lazily initializes and returns singleton DomainClassifierPredictor instance."""
    global _predictor
    if _predictor is None:
        from src.ml.predictor import DomainClassifierPredictor
        _predictor = DomainClassifierPredictor()
    return _predictor

def get_summarizer():
    """Lazily initializes and returns singleton DocumentSummarizer instance."""
    global _summarizer
    if _summarizer is None:
        from src.rag.summarizer import DocumentSummarizer
        _summarizer = DocumentSummarizer(vector_store=get_vector_store())
    return _summarizer

def get_comparator():
    """Lazily initializes and returns singleton DocumentComparator instance."""
    global _comparator
    if _comparator is None:
        from src.rag.comparator import DocumentComparator
        _comparator = DocumentComparator(vector_store=get_vector_store())
    return _comparator

def get_qa_chain():
    """Lazily initializes and returns singleton RAGQuestionAnswering instance."""
    global _qa_chain
    if _qa_chain is None:
        from src.rag.qa_chain import RAGQuestionAnswering
        _qa_chain = RAGQuestionAnswering(vector_store=get_vector_store())
    return _qa_chain

def get_analytics():
    """Lazily initializes and returns singleton SystemAnalytics instance."""
    global _analytics
    if _analytics is None:
        from src.analytics.metrics import SystemAnalytics
        _analytics = SystemAnalytics(vector_store=get_vector_store())
    return _analytics
