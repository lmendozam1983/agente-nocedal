FROM python:3.11-slim

WORKDIR /app

# Crear usuario no-root compatible con OpenShift
RUN useradd -m -u 1001 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ejecutar ingesta durante el build o asegurar permisos en la carpeta de datos
RUN python ingest.py || true
RUN chown -R 1001:0 /app && chmod -R g+rwX /app

USER 1001

EXPOSE 8080

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "main:app"]
