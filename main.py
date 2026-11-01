import os
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

app = FastAPI(title="Agente IA - Colegio Nocedal")

# Inicialización de Gemini y ChromaDB
API_KEY = os.getenv("GEMINI_API_KEY")
client_gemini = genai.Client(api_key=API_KEY)

embedding_fn = GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY,
    model_name="models/text-embedding-004"
)

chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_or_create_collection(
    name="nocedal_knowledge",
    embedding_function=embedding_fn
)

class ConsultaRequest(BaseModel):
    pregunta: str

@app.post("/chat")
def responder_consulta(req: ConsultaRequest):
    try:
        # 1. Recuperar contexto relevante de ChromaDB
        results = collection.query(query_texts=[req.pregunta], n_results=3)
        contexto = "\n\n".join(results['documents'][0]) if results['documents'] else "Sin información adicional."

        # 2. Construir el Prompt con restricción de contexto
        system_instruction = (
            "Eres el asistente virtual oficial de admisión del Colegio Nocedal de La Pintana. "
            "Responde a los apoderados con un tono amable, claro y empático. "
            "Usa EXCLUSIVAMENTE la información provista en el contexto. Si no está en el contexto, "
            "indica amablemente que pueden contactar al colegio a admision@colegionocedal.cl. "
            "Recuerda siempre mencionar que el colegio es 100% gratuito y que la postulación es vía SAE."
        )

        prompt = f"Contexto Oficial del Colegio:\n{contexto}\n\nPregunta del Apoderado: {req.pregunta}"

        # 3. Consultar a Gemini
        response = client_gemini.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={"system_instruction": system_instruction}
        )

        return {"respuesta": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
