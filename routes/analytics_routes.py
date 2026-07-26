from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.vector_store.manager import VectorStoreManager
from src.analytics.metrics import SystemAnalytics

router = APIRouter(prefix="/analytics", tags=["System Analytics"])

vector_store = VectorStoreManager()
analytics = SystemAnalytics(vector_store=vector_store)

@router.get("/stats")
def get_system_stats(db: Session = Depends(get_db)):
    """Computes overall usage, document indexing, TF category distribution, and query performance metrics."""
    return analytics.get_system_stats(db)

@router.get("/top-documents")
def get_top_documents(limit: int = 5, db: Session = Depends(get_db)):
    """Retrieves top documents ordered by chunk density and activity."""
    return analytics.get_top_referenced_documents(db, limit=limit)
