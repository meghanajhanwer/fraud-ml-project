FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]==0.30.*" "gunicorn==21.*"
COPY . .
ENV PORT=8080
CMD exec gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b :${PORT} app:app