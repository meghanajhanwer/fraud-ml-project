FROM python:3.12-slim
WORKDIR /app

# Optional: logs flush immediately
ENV PYTHONUNBUFFERED=1

# If you don't compile anything, drop build-essential to speed up
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]==0.30.*" "gunicorn==21.*"

COPY . .

# Cloud Run sets PORT at runtime
ENV PORT=8080

# Use gunicorn + uvicorn worker, and **bind to ${PORT}**
CMD exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b :${PORT} app:app