FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model during build (cached layer)
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy server
COPY server.py .

# Expose port
EXPOSE 8001

# Run server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
