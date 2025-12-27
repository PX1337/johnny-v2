FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server
COPY server.py .

# Expose port (Railway sets PORT dynamically)
EXPOSE ${PORT:-8001}

# Run server (model will be downloaded on first startup)
CMD sh -c "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8001}"
