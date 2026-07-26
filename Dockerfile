FROM python:3.10-slim

WORKDIR /app

# Prevent python from writing pyc files and buffer output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    CUDA_VISIBLE_DEVICES=""

# Install system build dependencies required for PyMuPDF & C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install lightweight python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Ensure runtime directories exist
RUN python -c "from config.settings import settings; settings.ensure_directories()"

# Expose port
EXPOSE 8000

# Run single worker Uvicorn server for 512MB RAM compliance
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
