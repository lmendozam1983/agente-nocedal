import os
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

def cargar_y_procesar_pdfs():
    # 1. Inicializar cliente de embeddings con Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    embedding_fn = GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )

    # 2. Conectar a ChromaDB persistente
    client = chromadb.PersistentClient(path="./chromadb_data")
    collection = client.get_or_create_collection(
        name="nocedal_knowledge",
        embedding_function=embedding_fn
    )

    # 3. Leer PDFs de la carpeta data/
    pdf_path = "data/admision_nocedal.pdf"
    if not os.path.exists(pdf_path):
        print("Agrega primero tus PDFs en la carpeta data/")
        return

    reader = PdfReader(pdf_path)
    textos, ids, metadatos = [], [], []

    chunk_id = 0
    for num_pagina, pagina in enumerate(reader.pages):
        texto_pagina = pagina.extract_text()
        if texto_pagina.strip():
            # Fragmentación básica por páginas/párrafos
            textos.append(texto_pagina)
            ids.append(f"doc_{chunk_id}")
            metadatos.append({"pagina": num_pagina + 1, "fuente": "admision_nocedal.pdf"})
            chunk_id += 1

    # 4. Guardar en la base de datos
    collection.add(documents=textos, ids=ids, metadatas=metadatos)
    print(f"Ingesta completada. {len(textos)} fragmentos guardados en ChromaDB.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    cargar_y_procesar_pdfs()
