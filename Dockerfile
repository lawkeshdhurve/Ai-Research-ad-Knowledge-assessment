FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for PyMuPDF and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure storage directories exist
RUN python -c "from config.settings import settings; settings.ensure_directories()"

# Expose FastAPI application port
EXPOSE 8000

# Run Uvicorn ASGI Production Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
