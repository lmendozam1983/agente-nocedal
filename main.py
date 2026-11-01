import os
from fastapi import FastAPI, HTTPException
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
        # Petición en lote para evitar múltiples llamadas HTTP lentas
        res = genai.embed_content(
            model=self.model_name,
            content=input,
            task_type="retrieval_query"
        )
        embeddings = res['embedding']
        # Si se envió un solo texto, encapsularlo en lista
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