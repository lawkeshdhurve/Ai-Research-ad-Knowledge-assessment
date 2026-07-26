import os
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.base import get_db
from src.database.models import DocumentMetadata
from src.document_processing.pdf_parser import PDFParser
from src.document_processing.chunker import DocumentChunker
from src.ml.predictor import DomainClassifierPredictor
from src.vector_store.manager import VectorStoreManager

router = APIRouter(prefix="/documents", tags=["Document Management"])

# Singletons for services
parser = PDFParser()
chunker = DocumentChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
predictor = DomainClassifierPredictor()
vector_store = VectorStoreManager()

def process_pdf_pipeline(doc_id: str, file_path: str, db_session_factory):
    """
    Background processing pipeline:
    1. Extract page text using PyMuPDF.
    2. Classify document domain using TensorFlow ML model.
    3. Create overlapping text chunks.
    4. Index embedded chunks in ChromaDB vector store.
    5. Update document status in SQLite metadata DB.
    """
    db: Session = db_session_factory()
    try:
        doc_record = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if not doc_record:
            return

        # Step 1: Text extraction
        pages_data = parser.extract_text_with_metadata(file_path, doc_id)
        if not pages_data:
            doc_record.processing_status = "FAILED"
            doc_record.error_message = "No readable text content extracted from PDF."
            db.commit()
            return

        total_pages = len(pages_data)
        
        # Step 2: TF Domain Classification on sample text
        sample_text = " ".join([p["text"] for p in pages_data[:3]])[:2000]
        prediction_result = predictor.predict_category(sample_text)
        predicted_category = prediction_result.get("category", "General Tech")

        # Step 3: Text Chunking
        chunks = chunker.create_chunks(pages_data)
        total_chunks = len(chunks)

        # Step 4: Vector Store Indexing
        vector_store.add_chunks(chunks)

        # Step 5: Update DB metadata record
        doc_record.total_pages = total_pages
        doc_record.total_chunks = total_chunks
        doc_record.category = predicted_category
        doc_record.processing_status = "PROCESSED"
        doc_record.error_message = None
        db.commit()

    except Exception as e:
        db.rollback()
        doc_record = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
        if doc_record:
            doc_record.processing_status = "FAILED"
            doc_record.error_message = str(e)
            db.commit()
        print(f"Pipeline processing error for doc {doc_id}: {e}")
    finally:
        db.close()


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """Uploads a PDF file and triggers async document processing pipeline."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(settings.RAW_DOCUMENTS_DIR, f"{doc_id}_{file.filename}")

    # Save uploaded file
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Initial DB record
    doc_record = DocumentMetadata(
        doc_id=doc_id,
        file_name=file.filename,
        file_path=file_path,
        processing_status="PENDING",
        category="Processing..."
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    # Trigger background pipeline
    from src.database.base import SessionLocal
    background_tasks.add_task(process_pdf_pipeline, doc_id, file_path, SessionLocal)

    return {
        "message": "Document uploaded successfully. Processing started.",
        "doc_id": doc_id,
        "file_name": file.filename,
        "status": "PENDING"
    }


@router.get("")
def list_documents(db: Session = Depends(get_db)):
    """Lists all uploaded documents with metadata, status, and TF category."""
    docs = db.query(DocumentMetadata).order_by(DocumentMetadata.upload_timestamp.desc()).all()
    return [d.to_dict() for d in docs]


@router.get("/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Fetches metadata for a specific document."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc.to_dict()


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """Deletes document from database, vector store, and filesystem storage."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove vector DB chunks
    vector_store.delete_document_chunks(doc_id)

    # Delete raw file if exists
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception as e:
            print(f"Error deleting file {doc.file_path}: {e}")

    # Remove database record
    db.delete(doc)
    db.commit()

    return {"message": "Document deleted successfully.", "doc_id": doc_id}


@router.post("/{doc_id}/reprocess")
def reprocess_document(
    doc_id: str, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    """Re-runs the ingestion, classification, chunking, and indexing pipeline."""
    doc = db.query(DocumentMetadata).filter(DocumentMetadata.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=400, detail="Raw document file missing on disk.")

    doc.processing_status = "PENDING"
    doc.error_message = None
    db.commit()

    # Clear old vectors
    vector_store.delete_document_chunks(doc_id)

    from src.database.base import SessionLocal
    background_tasks.add_task(process_pdf_pipeline, doc_id, doc.file_path, SessionLocal)

    return {"message": "Reprocessing started.", "doc_id": doc_id}
