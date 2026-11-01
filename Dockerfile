FROM python:3.11-slim

# Redirigir archivos temporales y HOME a /tmp
ENV HOME=/tmp
ENV TMPDIR=/tmp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear carpeta para ChromaDB y dar permisos completos a /app y /tmp
RUN mkdir -p /app/chromadb_data && chmod -R 777 /app /tmp

USER 1001

EXPOSE 8080

# Incluir --worker-tmp-dir /tmp para los procesos de Gunicorn
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--worker-tmp-dir", "/tmp", "--bind", "0.0.0.0:8080", "main:app"]
