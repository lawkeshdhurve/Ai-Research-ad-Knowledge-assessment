from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.services import get_analytics

router = APIRouter(prefix="/analytics", tags=["System Analytics"])

@router.get("/stats")
def get_system_stats(db: Session = Depends(get_db)):
    """Computes overall usage, document indexing, TF category distribution, and query performance metrics."""
    return get_analytics().get_system_stats(db)

@router.get("/top-documents")
def get_top_documents(limit: int = 5, db: Session = Depends(get_db)):
    """Retrieves top documents ordered by chunk density and activity."""
    return get_analytics().get_top_referenced_documents(db, limit=limit)
