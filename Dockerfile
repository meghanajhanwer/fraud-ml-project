FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8080
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir "uvicorn[standard]==0.30.*" "gunicorn==21.*"
COPY . .
CMD exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker \
    --timeout 90 --graceful-timeout 90 --keep-alive 75 \
    -b :${PORT} app:app
