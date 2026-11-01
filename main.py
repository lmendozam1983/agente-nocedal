import os
import glob
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
import google.generativeai as genai
from pypdf import PdfReader

app = FastAPI(title="Agente Colegio Nocedal")

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")


def embed_documents(textos: list[str]) -> list[list[float]]:
    """Embeddings para indexar documentos (en lotes de 100)."""
    vectores = []
    for i in range(0, len(textos), 100):
        lote = textos[i:i + 100]
        res = genai.embed_content(
            model=EMBED_MODEL,
            content=lote,
            task_type="retrieval_document",
        )
        vectores.extend(res["embedding"])
    return vectores


def embed_query(texto: str) -> list[float]:
    """Embedding para la pregunta del usuario."""
    res = genai.embed_content(
        model=EMBED_MODEL,
        content=texto,
        task_type="retrieval_query",
    )
    return res["embedding"]


chroma_client = chromadb.PersistentClient(path="./chromadb_data")

# Nombre nuevo para no chocar con la colección antigua (otra dimensión)
collection = chroma_client.get_or_create_collection(
    name="nocedal_info_v2",
    metadata={"hnsw:space": "cosine"},
)


def ingestar_pdf(pdf_path: str):
    """Extrae texto de un PDF e ingesta sus vectores en ChromaDB"""
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    chunk_size = 500
    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]

    if chunks:
        filename = os.path.basename(pdf_path)
        ids = [f"doc_{filename}_{i}" for i in range(len(chunks))]
        collection.upsert(
            documents=chunks,
            embeddings=embed_documents(chunks),
            ids=ids,
        )
    return len(chunks)


def cargar_todos_los_pdfs():
    """Ingesta todos los archivos .pdf que existan en la carpeta data/"""
    os.makedirs("data", exist_ok=True)
    pdf_files = glob.glob("data/*.pdf")
    total_chunks = 0
    for pdf in pdf_files:
        total_chunks += ingestar_pdf(pdf)
    return total_chunks


@app.on_event("startup")
def cargar_documentos_al_iniciar():
    try:
        if collection.count() == 0:
            cargar_todos_los_pdfs()
    except Exception as e:
        print(f"Error en startup: {e}")


class QueryRequest(BaseModel):
    pregunta: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/info")
def db_info():
    os.makedirs("data", exist_ok=True)
    archivos = os.listdir("data")
    return {
        "documentos_en_db": collection.count(),
        "archivos_en_data": archivos,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos en formato .pdf")

    os.makedirs("data", exist_ok=True)
    # basename evita rutas maliciosas tipo ../../archivo.pdf
    file_path = os.path.join("data", os.path.basename(file.filename))

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_cargados = ingestar_pdf(file_path)
        return {
            "mensaje": f"Archivo '{file.filename}' subido e ingestado con éxito.",
            "fragmentos": chunks_cargados,
            "total_documentos": collection.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Asistente Virtual - Colegio Nocedal</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 h-screen flex flex-col items-center justify-center p-4">
        <div class="w-full max-w-2xl bg-white rounded-xl shadow-lg flex flex-col h-[650px] overflow-hidden border border-gray-200">
            <!-- Encabezado con opción de carga -->
            <div class="bg-blue-900 text-white p-4 flex items-center justify-between">
                <div>
                    <h1 class="font-bold text-lg">🤖 Agente IA - Colegio Nocedal</h1>
                    <p class="text-xs text-blue-200">Base de conocimiento institucional</p>
                </div>
                <label class="cursor-pointer bg-blue-700 hover:bg-blue-600 text-xs px-3 py-2 rounded-lg font-semibold transition-colors flex items-center gap-1">
                    📁 Subir PDF
                    <input type="file" id="pdf-file" accept=".pdf" class="hidden" onchange="uploadPDF()">
                </label>
            </div>

            <!-- Estado de la subida -->
            <div id="upload-status" class="hidden bg-yellow-100 text-yellow-800 text-xs px-4 py-2 border-b"></div>

            <!-- Historial del Chat -->
            <div id="chat-box" class="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-50 flex flex-col">
                <div class="bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[80%] self-start shadow-sm">
                    ¡Hola! Soy el asistente institucional del Colegio Nocedal. Puedes realizar tus consultas sobre la admisión o cargar nueva documentación usando el botón superior.
                </div>
            </div>

            <!-- Entrada de texto -->
            <div class="p-4 border-t flex gap-2 bg-white">
                <input id="user-input" type="text" placeholder="Escribe tu consulta aquí..." class="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600" onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" class="bg-blue-900 text-white px-5 py-2 rounded-lg hover:bg-blue-800 font-semibold transition-colors">Enviar</button>
            </div>
        </div>

        <script>
            function addMessage(text, classes) {
                const chatBox = document.getElementById('chat-box');
                const div = document.createElement('div');
                div.className = classes + ' whitespace-pre-wrap';
                div.textContent = text;  // textContent evita inyección de HTML
                chatBox.appendChild(div);
                chatBox.scrollTop = chatBox.scrollHeight;
                return div;
            }

            async function uploadPDF() {
                const fileInput = document.getElementById('pdf-file');
                const statusDiv = document.getElementById('upload-status');
                if (!fileInput.files.length) return;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                statusDiv.className = "bg-yellow-100 text-yellow-800 text-xs px-4 py-2 border-b";
                statusDiv.innerText = `Subiendo e ingestando '${fileInput.files[0].name}'... Espere un momento.`;

                try {
                    const res = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        statusDiv.className = "bg-green-100 text-green-800 text-xs px-4 py-2 border-b";
                        statusDiv.innerText = `${data.mensaje} Total fragmentos en base de datos: ${data.total_documentos}`;
                    } else {
                        statusDiv.className = "bg-red-100 text-red-800 text-xs px-4 py-2 border-b";
                        statusDiv.innerText = `Error: ${data.detail || 'Fallo al subir el archivo.'}`;
                    }
                } catch (e) {
                    statusDiv.className = "bg-red-100 text-red-800 text-xs px-4 py-2 border-b";
                    statusDiv.innerText = "Error de conexión al subir archivo.";
                }
                fileInput.value = '';
            }

            async function sendMessage() {
                const input = document.getElementById('user-input');
                const text = input.value.trim();
                if (!text) return;

                addMessage(text, "bg-blue-900 text-white p-3 rounded-lg max-w-[80%] self-end ml-auto shadow-sm");
                input.value = '';

                const loadingElem = addMessage("Consultando información...", "bg-gray-200 text-gray-600 p-3 rounded-lg max-w-[80%] self-start italic");

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pregunta: text })
                    });

                    const data = await response.json();
                    loadingElem.remove();
                    addMessage(data.respuesta || data.detail || 'Sin respuesta.', "bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[80%] self-start shadow-sm");
                } catch (e) {
                    loadingElem.remove();
                    addMessage("Error de conexión.", "bg-red-100 text-red-800 p-3 rounded-lg max-w-[80%] self-start shadow-sm");
                }
            }
        </script>
    </body>
    </html>
    """


@app.post("/chat")
def chat(request: QueryRequest):
    if not request.pregunta.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    try:
        if collection.count() == 0:
            cargar_todos_los_pdfs()

        count = collection.count()
        if count == 0:
            contexto = "No hay información disponible en la base de conocimientos."
        else:
            results = collection.query(
                query_embeddings=[embed_query(request.pregunta)],
                n_results=min(3, count),
            )
            context_docs = results.get("documents", [[]])[0]
            contexto = "\n---\n".join(context_docs) if context_docs else "Sin contexto relevante."

        prompt = f"""Eres un asistente amigable de admisión del Colegio Nocedal (La Pintana, Santiago).
Responde la pregunta del usuario utilizando la siguiente información institucional:

CONTEXTO INSTITUCIONAL:
{contexto}

PREGUNTA DEL USUARIO:
{request.pregunta}
"""
        model = genai.GenerativeModel(CHAT_MODEL)
        response = model.generate_content(prompt)
        return {"respuesta": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")