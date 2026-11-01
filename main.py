import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai

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
        embeddings = []
        for text in input:
            res = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )
            embeddings.append(res['embedding'])
        return embeddings

chroma_client = chromadb.PersistentClient(path="./chromadb_data")
embedding_fn = GeminiEmbeddingFunction(api_key=api_key)

collection = chroma_client.get_or_create_collection(
    name="nocedal_info",
    embedding_function=embedding_fn
)

class QueryRequest(BaseModel):
    pregunta: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: QueryRequest):
    if not request.pregunta:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")
    
    results = collection.query(
        query_texts=[request.pregunta],
        n_results=3
    )
    
    context_docs = results.get("documents", [[]])[0]
    contexto = "\n---\n".join(context_docs) if context_docs else "No se encontró información adicional."
    
    prompt = f"""Eres un asistente amigable de admisión del Colegio Nocedal (La Pintana, Santiago).
Responde la pregunta del usuario utilizando la siguiente información del colegio:

CONTEXTO:
{contexto}

PREGUNTA:
{request.pregunta}
"""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return {"respuesta": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar respuesta: {str(e)}")
