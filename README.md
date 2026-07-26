# AI Research & Knowledge Assistant

An enterprise-ready, production-grade **AI Research & Knowledge Assistant** designed to ingest, classify, index, query, summarize, and compare technical research papers, specifications, and whitepapers.

---

## 🌟 Key Capabilities

- **PDF Ingestion & Page Metadata**: Ingests multi-page PDFs, preserving exact page numbers, metadata, and background processing status tracking (`PENDING`, `PROCESSED`, `FAILED`).
- **TensorFlow ML Domain Classifier**: Deep learning model built with TensorFlow/Keras to auto-categorize uploaded documents into technology domains (*Artificial Intelligence*, *Cyber Security*, *Cloud Computing*, *Robotics*, *Data Engineering*).
- **Dense & Hybrid Vector Search**: Persistent ChromaDB vector database with `sentence-transformers/all-MiniLM-L6-v2` embeddings, providing dense semantic similarity, sparse keyword matching, and Reciprocal Rank Fusion (RRF) hybrid search.
- **Citation-Grounded RAG Engine**: Generates natural language answers strictly grounded in document context, complete with explicit file and page-number citations (`[Document_Name.pdf - Page 3]`).
- **Conversational Memory**: Session-based chat history for multi-turn research conversations.
- **Multi-Document Comparison & Summarization**: Structured side-by-side matrices comparing research methodologies, advantages, limitations, and multi-tier summaries (Executive, Technical, Key Takeaways).
- **System Analytics**: Real-time performance tracking for document index counts, total chunks, query execution latency, and TF category distributions.
- **Modern Web Interface**: Premium dark-mode glassmorphism dashboard with drag-and-drop file upload, inline citation badges, interactive RAG chat, multi-doc analysis, analytics cards, and ML sandbox.

---

## 🏗️ Architecture

```
                               ┌────────────────┐
                               │ PDF Upload     │
                               └───────┬────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │ PyMuPDF Parser & Metadata Extractor   │
                   └───────┬───────────────────────┬───────┘
                           │                       │
                           ▼                       ▼
           ┌──────────────────────────────┐  ┌─────────────────────────┐
           │ Recursive Text Chunker       │  │ TensorFlow Domain       │
           │ (800-1000 chars, 120 overlap) │  │ Classifier (.h5 Model)  │
           └───────────────┬──────────────┘  └─────────────────────────┘
                           │
                           ▼
           ┌──────────────────────────────┐
           │ SentenceTransformer Embedding│
           └───────────────┬──────────────┘
                           │
                           ▼
           ┌──────────────────────────────┐
           │ ChromaDB Vector Index         │
           └───────────────┬──────────────┘
                           │
                           ▼
           ┌──────────────────────────────┐      ┌─────────────────────────┐
           │ Hybrid Vector & RRF Retriever │ ───► │ RAG QA + Page Citations │
           └──────────────────────────────┘      └─────────────────────────┘
```

---

## 📁 Repository Structure

```
ai-research-assistant/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Application settings & storage paths (Pydantic)
│
├── data/
│   ├── raw_documents/           # Stored uploaded PDF files
│   ├── vector_db/               # Persistent ChromaDB vector storage
│   └── dataset/                 # Training dataset artifacts
│
├── models/
│   ├── tf_classifier.h5         # Saved TensorFlow trained model
│   └── tokenizer.pickle         # Saved tokenizer / vectorizer artifact
│
├── src/
│   ├── database/                # SQLite ORM models & session setup
│   ├── document_processing/     # PyMuPDF text parser & recursive chunker
│   ├── ml/                      # TensorFlow dataset prep, model training & predictor
│   ├── vector_store/            # ChromaDB hybrid search vector manager
│   ├── rag/                     # RAG QA chain with citations, summarizer, comparator
│   └── analytics/               # System metrics computation
│
├── routes/                      # FastAPI REST API endpoints
│   ├── document_routes.py
│   ├── search_routes.py
│   ├── analysis_routes.py
│   └── analytics_routes.py
│
├── static/                      # Web UI Dashboard (HTML, CSS, JS)
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── tests/                       # Pytest automated test suite
│   ├── test_parser.py
│   ├── test_rag.py
│   └── test_ml.py
│
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- **Python**: `3.10` or `3.11`
- **RAM**: Minimum 8 GB (16 GB recommended)
- **Disk Space**: 5 GB

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Backend Server

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Dashboard**: Access `http://localhost:8000` in your browser.
- **Swagger API Documentation**: Access `http://localhost:8000/docs`.

---

## 🧪 Running Automated Tests

Execute the test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## ⚡ REST API Summary

| Category | Endpoint | Method | Description |
| --- | --- | --- | --- |
| **Documents** | `/documents/upload` | `POST` | Upload PDF & start async ingestion pipeline |
| **Documents** | `/documents` | `GET` | List all uploaded documents with TF domain category |
| **Documents** | `/documents/{doc_id}` | `DELETE` | Delete document from DB, storage & vector index |
| **Search** | `/search/semantic` | `POST` | Execute dense, keyword, or hybrid RRF retrieval |
| **Search** | `/search/qa` | `POST` | Grounded RAG query with page citations & session memory |
| **Analysis** | `/analysis/summarize` | `POST` | Generate Executive & Technical multi-tier summary |
| **Analysis** | `/analysis/compare` | `POST` | Side-by-side comparative matrix of 2+ documents |
| **Analysis** | `/analysis/classify` | `POST` | Predict text domain using TensorFlow neural network |
| **Analytics** | `/analytics/stats` | `GET` | System index stats, latency, & category breakdown |
