import os
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import google.generativeai as genai
from pypdf import PdfReader

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
                task_type="retrieval_document"
            )
            embeddings.append(res['embedding'])
        return embeddings

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Aviso: GEMINI_API_KEY no configurada. Omitiendo ingesta previa.")
        return

    pdf_path = "data/admision_nocedal.pdf"
    if not os.path.exists(pdf_path):
        print(f"Aviso: {pdf_path} no existe. Omitiendo ingesta.")
        return

    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]

    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    embedding_fn = GeminiEmbeddingFunction(api_key=api_key)
    
    collection = chroma_client.get_or_create_collection(
        name="nocedal_info",
        embedding_function=embedding_fn
    )

    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids
    )
    print(f"Ingesta exitosa: {len(chunks)} fragmentos procesados.")

if __name__ == "__main__":
    main()