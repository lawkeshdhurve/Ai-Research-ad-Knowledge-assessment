from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.vector_store.manager import VectorStoreManager
from src.rag.summarizer import DocumentSummarizer
from src.rag.comparator import DocumentComparator
from src.ml.predictor import DomainClassifierPredictor

router = APIRouter(prefix="/analysis", tags=["Document Analysis & ML Classification"])

vector_store = VectorStoreManager()
summarizer = DocumentSummarizer(vector_store=vector_store)
comparator = DocumentComparator(vector_store=vector_store)
predictor = DomainClassifierPredictor()

class SummarizeRequest(BaseModel):
    doc_id: str

class CompareRequest(BaseModel):
    doc_ids: List[str]

class ClassifyRequest(BaseModel):
    text: str


@router.post("/summarize")
def summarize_document(req: SummarizeRequest, db: Session = Depends(get_db)):
    """Generates structured multi-tier document summary."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == req.doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    file_name = doc.file_name if doc else "Document"
    result = summarizer.summarize_document(req.doc_id, file_name=file_name)
    return result


@router.post("/compare")
def compare_documents(req: CompareRequest, db: Session = Depends(get_db)):
    """Compares 2 or more selected documents across methodologies, pros/cons, similarities, differences."""
    if len(req.doc_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 document IDs are required for comparison.")

    docs = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id.in_(req.doc_ids)).all()
    doc_map = {d.doc_id: d.file_name for d in docs}
    names = [doc_map.get(d_id, f"Doc_{d_id[:6]}") for d_id in req.doc_ids]

    result = comparator.compare_documents(doc_ids=req.doc_ids, doc_names=names)
    return result


@router.post("/classify")
def classify_text(req: ClassifyRequest):
    """Predicts domain classification using saved TensorFlow neural network model."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")

    prediction = predictor.predict_category(req.text)
    return prediction
