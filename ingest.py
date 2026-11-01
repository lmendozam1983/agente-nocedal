import os
import glob
import chromadb
import google.generativeai as genai
from pypdf import PdfReader

EMBED_MODEL = "models/gemini-embedding-001"
COLLECTION_NAME = "nocedal_info_v2"


def embed_documents(textos: list[str]) -> list[list[float]]:
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


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Aviso: GEMINI_API_KEY no configurada. Omitiendo ingesta previa.")
        return
    genai.configure(api_key=api_key)

    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("Aviso: no hay PDFs en data/. Omitiendo ingesta.")
        return

    chroma_client = chromadb.PersistentClient(path="./chromadb_data")
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = 0
    for pdf_path in pdf_files:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)

        chunk_size = 500
        chunks = [
            text[i:i + chunk_size]
            for i in range(0, len(text), chunk_size)
            if text[i:i + chunk_size].strip()
        ]
        if not chunks:
            continue

        filename = os.path.basename(pdf_path)
        ids = [f"doc_{filename}_{i}" for i in range(len(chunks))]
        collection.upsert(
            documents=chunks,
            embeddings=embed_documents(chunks),
            ids=ids,
        )
        total += len(chunks)

    print(f"Ingesta exitosa: {total} fragmentos procesados.")


if __name__ == "__main__":
    main()