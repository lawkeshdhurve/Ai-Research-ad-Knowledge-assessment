from typing import Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.database.models import DocumentMetadata, ChatMessage, QueryMetric
from src.vector_store.manager import VectorStoreManager

class SystemAnalytics:
    """Analytics computation engine for document, index, and query performance statistics."""

    def __init__(self, vector_store: VectorStoreManager = None):
        self.vector_store = vector_store or VectorStoreManager()

    def get_system_stats(self, db: Session) -> Dict[str, Any]:
        """Calculates system metrics across documents, chunks, categories, and queries."""
        # 1. Total Documents & Status Breakdown
        total_documents = db.query(DocumentMetadata).count()
        processed_docs = db.query(DocumentMetadata).filter(DocumentMetadata.processing_status == "PROCESSED").count()
        pending_docs = db.query(DocumentMetadata).filter(DocumentMetadata.processing_status == "PENDING").count()
        failed_docs = db.query(DocumentMetadata).filter(DocumentMetadata.processing_status == "FAILED").count()

        # 2. Total Vector Chunks
        total_chunks = self.vector_store.get_total_chunk_count()

        # 3. Domain Category Distribution
        category_counts_query = (
            db.query(DocumentMetadata.category, func.count(DocumentMetadata.doc_id))
            .group_by(DocumentMetadata.category)
            .all()
        )
        category_distribution = {cat: count for cat, count in category_counts_query if cat}

        # 4. Total Queries & Performance Stats
        total_queries = db.query(QueryMetric).count()
        avg_execution_time = (
            db.query(func.avg(QueryMetric.execution_time_ms)).scalar() or 0.0
        )

        return {
            "total_documents": total_documents,
            "processed_documents": processed_docs,
            "pending_documents": pending_docs,
            "failed_documents": failed_docs,
            "total_indexed_chunks": total_chunks,
            "category_distribution": category_distribution,
            "total_queries_executed": total_queries,
            "average_latency_ms": round(float(avg_execution_time), 2)
        }

    def get_top_referenced_documents(self, db: Session, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns top documents by chunk count and query activity."""
        docs = db.query(DocumentMetadata).order_by(DocumentMetadata.total_chunks.desc()).limit(limit).all()
        return [
            {
                "doc_id": d.doc_id,
                "file_name": d.file_name,
                "category": d.category,
                "total_chunks": d.total_chunks,
                "total_pages": d.total_pages,
                "upload_timestamp": d.upload_timestamp.isoformat() if d.upload_timestamp else None
            }
            for d in docs
        ]
