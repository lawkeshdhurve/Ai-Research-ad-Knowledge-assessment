import datetime
import json
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from src.database.base import Base

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    doc_id = Column(String(64), primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processing_status = Column(String(32), default="PENDING")  # PENDING, PROCESSED, FAILED
    category = Column(String(128), default="General Tech")
    error_message = Column(Text, nullable=True)

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "processing_status": self.processing_status,
            "category": self.category,
            "error_message": self.error_message
        }


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String(64), primary_key=True, index=True)
    title = Column(String(255), default="New Research Session")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), ForeignKey("chat_sessions.session_id"), nullable=False)
    role = Column(String(32), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    citations_json = Column(Text, default="[]")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

    def citations(self):
        try:
            return json.loads(self.citations_json) if self.citations_json else []
        except Exception:
            return []


class QueryMetric(Base):
    __tablename__ = "query_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    retrieved_doc_ids_json = Column(Text, default="[]")
    execution_time_ms = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
