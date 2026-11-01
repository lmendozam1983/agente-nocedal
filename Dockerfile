FROM python:3.11-slim

ENV HOME=/tmp
ENV TMPDIR=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/chromadb_data && chmod -R 777 /app /tmp

USER 1001

EXPOSE 8080

# Se usa -w 1 para evitar bloqueos de archivo SQLite en ChromaDB
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--worker-tmp-dir", "/tmp", "--timeout", "120", "--bind", "0.0.0.0:8080", "main:app"]
