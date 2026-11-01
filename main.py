import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai
from pypdf import PdfReader

app = FastAPI(title="Agente Colegio Nocedal")

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str = None, model_name: str = "models/embedding-001"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        res = genai.embed_content(
            model=self.model_name,
            content=input,
            task_type="retrieval_query"
        )
        embeddings = res['embedding']
        if isinstance(embeddings[0], (float, int)):
            return [embeddings]
        return embeddings

chroma_client = chromadb.PersistentClient(path="./chromadb_data")
embedding_fn = GeminiEmbeddingFunction(api_key=api_key)

collection = chroma_client.get_or_create_collection(
    name="nocedal_info",
    embedding_function=embedding_fn
)

def cargar_documentos_si_vacio():
    try:
        if collection.count() == 0:
            pdf_path = "data/admision_nocedal.pdf"
            if os.path.exists(pdf_path):
                print("Iniciando ingesta de documentos...")
                reader = PdfReader(pdf_path)
                text = "".join([page.extract_text() or "" for page in reader.pages])
                chunk_size = 500
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
                if chunks:
                    ids = [f"doc_{i}" for i in range(len(chunks))]
                    collection.add(documents=chunks, ids=ids)
                    print(f"Ingesta exitosa: {len(chunks)} fragmentos cargados.")
    except Exception as e:
        print(f"Error durante la auto-ingesta: {e}")

cargar_documentos_si_vacio()

class QueryRequest(BaseModel):
    pregunta: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- INTERFAZ GRÁFICA EN RUTA RAÍZ ---
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
        <div class="w-full max-w-2xl bg-white rounded-xl shadow-lg flex flex-col h-[600px] overflow-hidden border border-gray-200">
            <!-- Encabezado -->
            <div class="bg-blue-900 text-white p-4 font-bold text-lg flex items-center justify-between">
                <span>🤖 Agente IA - Colegio Nocedal</span>
                <span class="text-xs bg-blue-700 px-2 py-1 rounded">En línea</span>
            </div>
            <!-- Historial de Chat -->
            <div id="chat-box" class="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-50 flex flex-col">
                <div class="bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[80%] self-start shadow-sm">
                    ¡Hola! Soy el asistente institucional del Colegio Nocedal. Puedes preguntarme sobre la admisión, matrícula o especialidades técnico-profesionales.
                </div>
            </div>
            <!-- Entrada de usuario -->
            <div class="p-4 border-t flex gap-2 bg-white">
                <input id="user-input" type="text" placeholder="Escribe tu mensaje aquí..." class="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-600" onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()" class="bg-blue-900 text-white px-5 py-2 rounded-lg hover:bg-blue-800 font-semibold transition-colors">Enviar</button>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const text = input.value.trim();
                if (!text) return;

                // Agregar mensaje de usuario
                chatBox.innerHTML += `<div class="bg-blue-900 text-white p-3 rounded-lg max-w-[80%] self-end ml-auto shadow-sm">${text}</div>`;
                input.value = '';
                chatBox.scrollTop = chatBox.scrollHeight;

                // Indicador de carga
                const loadingId = 'loading-' + Date.now();
                chatBox.innerHTML += `<div id="${loadingId}" class="bg-gray-200 text-gray-600 p-3 rounded-lg max-w-[80%] self-start italic">Consultando información...</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;

                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ pregunta: text })
                    });
                    const data = await response.json();
                    document.getElementById(loadingId).remove();

                    const respuestaText = data.respuesta || data.detail || 'Ocurrió un error inesperado.';
                    chatBox.innerHTML += `<div class="bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[80%] self-start shadow-sm">${respuestaText.replace(/\\n/g, '<br>')}</div>`;
                } catch (e) {
                    document.getElementById(loadingId).remove();
                    chatBox.innerHTML += `<div class="bg-red-100 text-red-800 p-3 rounded-lg max-w-[80%] self-start shadow-sm">Error de conexión con la API.</div>`;
                }
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    </body>
    </html>
    """

@app.post("/chat")
def chat(request: QueryRequest):
    if not request.pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    if collection.count() == 0:
        contexto = "No hay documentos adicionales disponibles en la base de conocimientos."
    else:
        results = collection.query(
            query_texts=[request.pregunta],
            n_results=min(3, collection.count())
        )
        context_docs = results.get("documents", [[]])[0]
        contexto = "\n---\n".join(context_docs) if context_docs else "Sin contexto relevante."
    
    prompt = f"""Eres un asistente amigable de admisión del Colegio Nocedal (La Pintana, Santiago).
Responde la pregunta del usuario utilizando la siguiente información institucional:

CONTEXTO:
{contexto}

PREGUNTA:
{request.pregunta}
"""
    
    try:
        model = genai.GenerativeModel("gemini-3.8-flash")
        response = model.generate_content(prompt)
        return {"respuesta": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {str(e)}")