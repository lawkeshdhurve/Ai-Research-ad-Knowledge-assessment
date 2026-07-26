import time
import uuid
import json
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.database.base import get_db
from src.database.models import ChatSession, ChatMessage, QueryMetric
from src.services import get_vector_store, get_qa_chain

# Pydantic Request Models
class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"  # "semantic", "keyword", "hybrid"
    top_k: int = 4
    doc_ids: Optional[List[str]] = None

class QARequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Research Session"


@router.post("/semantic")
def search_documents(req: SearchRequest):
    """Executes dense semantic, sparse keyword, or hybrid RRF retrieval."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    vector_store = get_vector_store()
    if req.search_type == "semantic":
        results = vector_store.semantic_search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
    elif req.search_type == "keyword":
        results = vector_store.keyword_search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)
    else:
        results = vector_store.hybrid_search(req.query, top_k=req.top_k, doc_ids=req.doc_ids)

    return {
        "query": req.query,
        "search_type": req.search_type,
        "results_count": len(results),
        "results": results
    }


@router.post("/qa")
def answer_question(req: QARequest, db: Session = Depends(get_db)):
    """
    RAG QA Endpoint: retrieves context, grounded answer, page citations, 
    manages multi-turn conversation memory, and logs query metrics.
    """
    start_time = time.time()

    # Session management
    session_id = req.session_id
    session_record = None

    if session_id:
        session_record = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()

    if not session_record:
        session_id = str(uuid.uuid4())
        session_record = ChatSession(session_id=session_id, title=req.query[:40])
        db.add(session_record)
        db.commit()

    # Fetch conversation history
    past_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    history_payload = [{"role": m.role, "content": m.content} for m in past_messages]

    # Save user query message
    user_msg = ChatMessage(session_id=session_id, role="user", content=req.query)
    db.add(user_msg)
    db.commit()

    # Run RAG QA chain
    qa_result = get_qa_chain().answer_question(
        query=req.query,
        session_history=history_payload,
        selected_doc_ids=req.doc_ids
    )

    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    # Save assistant message with citations
    assistant_msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=qa_result["answer"],
        citations_json=json.dumps(qa_result["citations"])
    )
    db.add(assistant_msg)

    # Log query metric
    retrieved_ids = list(set([c["doc_id"] for c in qa_result["citations"] if c.get("doc_id")]))
    metric_record = QueryMetric(
        query_text=req.query,
        retrieved_doc_ids_json=json.dumps(retrieved_ids),
        execution_time_ms=execution_time_ms
    )
    db.add(metric_record)
    db.commit()

    return {
        "session_id": session_id,
        "query": req.query,
        "answer": qa_result["answer"],
        "citations": qa_result["citations"],
        "retrieved_context": qa_result["retrieved_context"],
        "execution_time_ms": execution_time_ms
    }


@router.post("/sessions")
def create_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Creates a new chat session."""
    s_id = str(uuid.uuid4())
    session = ChatSession(session_id=s_id, title=req.title or "New Research Session")
    db.add(session)
    db.commit()
    return {"session_id": s_id, "title": session.title}


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)):
    """Lists all active chat sessions."""
    sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
    return [
        {
            "session_id": s.session_id,
            "title": s.title,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat() if s.updated_at else None
        }
        for s in sessions
    ]


@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """Fetches chat message history for a specific session."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "citations": m.citations(),
            "timestamp": m.timestamp.isoformat()
        }
        for m in messages
    ]
