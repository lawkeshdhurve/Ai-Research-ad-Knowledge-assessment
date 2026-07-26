import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config.settings import settings
from src.database.base import init_db
from routes import document_routes, search_routes, analysis_routes, analytics_routes

# Initialize Database Tables
init_db()

# Create FastAPI Instance
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise-grade AI Research & Knowledge Assistant for PDF ingestion, RAG QA, TF domain classification, and analytics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers under /api/v1
api_prefix = settings.API_V1_STR
app.include_router(document_routes.router, prefix=api_prefix)
app.include_router(search_routes.router, prefix=api_prefix)
app.include_router(analysis_routes.router, prefix=api_prefix)
app.include_router(analytics_routes.router, prefix=api_prefix)

# Also register at root level for direct endpoint access
app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(analysis_routes.router)
app.include_router(analytics_routes.router)

# Static Files Directory Setup
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def read_root():
    """Serves the main Web Interface dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({
        "app": settings.APP_NAME,
        "status": "Running",
        "docs": "/docs"
    })


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": settings.APP_NAME}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
