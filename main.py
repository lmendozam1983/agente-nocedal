import os
import glob
import shutil
import asyncio
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from pypdf import PdfReader

app = FastAPI(title="Agente Colegio Nocedal - RIE High Performance")

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

chroma_client = chromadb.PersistentClient(path="./chromadb_data")
collection = chroma_client.get_or_create_collection(
    name="nocedal_info_local",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

def dividir_en_chunks(texto: str) -> list[str]:
    chunks = []
    paso = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(texto), paso):
        fragmento = texto[i:i + CHUNK_SIZE]
        if fragmento.strip():
            chunks.append(fragmento)
    return chunks

def ingestar_pdf(pdf_path: str):
    reader = PdfReader(pdf_path)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    chunks = dividir_en_chunks(text)

    if chunks:
        filename = os.path.basename(pdf_path)
        ids = [f"doc_{filename}_{i}" for i in range(len(chunks))]
        collection.upsert(documents=chunks, ids=ids)
    return len(chunks)

def cargar_todos_los_pdfs():
    os.makedirs("data", exist_ok=True)
    pdf_files = glob.glob("data/*.pdf")
    total_chunks = 0
    for pdf in pdf_files:
        try:
            total_chunks += ingestar_pdf(pdf)
        except Exception as e:
            print(f"Error procesando {pdf}: {e}")
    return total_chunks

@app.on_event("startup")
def cargar_documentos_al_iniciar():
    if collection.count() == 0:
        hilo_ingesta = threading.Thread(target=cargar_todos_los_pdfs)
        hilo_ingesta.start()

class QueryRequest(BaseModel):
    pregunta: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/info")
def db_info():
    os.makedirs("data", exist_ok=True)
    return {
        "documentos_en_db": collection.count(),
        "archivos_en_data": os.listdir("data"),
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .pdf")

    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", os.path.basename(file.filename))

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_cargados = ingestar_pdf(file_path)
        return {
            "mensaje": f"Archivo '{file.filename}' ingestado con éxito.",
            "fragmentos": chunks_cargados,
            "total_documentos": collection.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {str(e)}")

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Asistente Virtual - Colegio Nocedal</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body class="bg-slate-100 h-screen flex flex-col items-center justify-center p-2 sm:p-4">
    <div class="w-full max-w-2xl bg-white rounded-xl shadow-xl flex flex-col h-[680px] overflow-hidden border border-slate-200">
        <!-- Encabezado -->
        <div class="bg-blue-900 text-white p-4 flex items-center justify-between">
            <div>
                <h1 class="font-bold text-lg">🤖 Agente IA - Colegio Nocedal</h1>
                <p class="text-xs text-blue-200">Base de conocimiento RIE - Ultra Rápido</p>
            </div>
            <label class="cursor-pointer bg-blue-700 hover:bg-blue-600 text-xs px-3 py-2 rounded-lg font-semibold transition-colors flex items-center gap-1">
                📁 Subir PDF
                <input type="file" id="pdf-file" accept=".pdf" class="hidden" onchange="uploadPDF()">
            </label>
        </div>

        <div id="upload-status" class="hidden bg-yellow-100 text-yellow-800 text-xs px-4 py-2 border-b"></div>

        <!-- Chat Box -->
        <div id="chat-box" class="flex-1 p-4 overflow-y-auto space-y-4 bg-slate-50 flex flex-col">
            <div class="bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[85%] self-start shadow-sm text-sm">
                ¡Hola! Bienvenid@ al asistente institucional del Colegio Nocedal. Escribe tu consulta sobre el reglamento o selecciona una opción rápida.
            </div>
        </div>

        <!-- Menú de Atajos Rápidos -->
        <div class="px-4 py-2 bg-slate-100 border-t flex gap-2 overflow-x-auto text-xs">
            <button onclick="sendQuickAction('📞 Contacto Directo')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                📞 Contacto
            </button>
            <button onclick="sendQuickAction('📍 Horarios')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                📍 Horarios
            </button>
            <button onclick="sendQuickAction('📋 Matrículas y Vacantes')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                📋 Matrículas
            </button>
            <button onclick="sendQuickAction('👔 Uniforme y Presentación')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                👔 Uniforme
            </button>
            <button onclick="sendQuickAction('📱 Uso de Celulares')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                📱 Celulares
            </button>
            <button onclick="sendQuickAction('📋 Director y Equipo')" class="bg-white hover:bg-blue-50 text-blue-900 font-medium px-3 py-1.5 rounded-full border border-blue-200 shadow-sm whitespace-nowrap transition-colors">
                👔 Directores
            </button>
        </div>

        <!-- Entrada de Texto -->
        <div class="p-3 border-t flex gap-2 bg-white">
            <input id="user-input" type="text" placeholder="Escribe tu consulta aquí..." class="flex-1 border rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600" onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()" class="bg-blue-900 text-white px-5 py-2 rounded-lg hover:bg-blue-800 font-semibold text-sm transition-colors">Enviar</button>
        </div>
    </div>

    <script>
        const FAST_RESPONSES = {
            '📞 Contacto Directo': '**📌 Contactos Oficiales - Colegio Nocedal**\\n\\n* **Teléfono Secretaría:** +56 2 2543 3816\\n* **Correo Admisión:** admision@nocedal.cl\\n* **Sitio Web:** www.nocedal.cl',
            '📍 Horarios': '**📌 Horarios Oficiales de Funcionamiento**\\n\\n* **Apertura del colegio:** 07:30 hrs.\\n* **Entrada general:** 08:30 hrs.\\n* **Salida Básica (1° a 6°):** 16:00 hrs.\\n* **Salida Media (I° a IV°):** Entre 15:55 y 17:35 hrs según el día.',
            '📋 Matrículas y Vacantes': '**📌 Proceso de Admisión y Matrículas**\\n\\n* **Sistema Oficial:** La postulación a vacantes se realiza únicamente a través del **Sistema de Admisión Escolar (SAE)**.\\n* **Consultas Presenciales:** En secretaría de Lunes a Viernes de 08:30 a 17:00 hrs.\\n* **Correo Admisión:** admision@nocedal.cl',
            '👔 Uniforme y Presentación': '**📌 Reglamento de Uniforme y Presentación**\\n\\n* **Uniforme Oficial:** Pantalón gris, camisa blanca, corbata institucional, chaleco azul marino con insignia y zapatos negros.\\n* **Corte de pelo:** Tradicional escolar sobrio (corto a los lados y atrás), sin teñidos ni barba.',
            '📱 Uso de Celulares': '**📌 Normativa sobre Dispositivos Electrónicos**\\n\\n* Prohibido el uso de celulares durante las clases sin autorización docente.\\n* En caso de uso no autorizado, el equipo es retirado por Coordinación hasta el fin de la jornada.',
            '📋 Director y Equipo': '**📌 Equipo Directivo - Colegio Nocedal**\\n\\n* **Director de Ciclo:** Instancia superior de resolución de asuntos académicos y de convivencia escolar.\\n* **Conducto Regular:** Profesor de Asignatura ➔ Profesor Jefe ➔ Coordinador de Nivel ➔ Director de Ciclo.'
        };

        function renderMarkdown(text) {
            if (!text) return '';
            const unescapedText = text.replace(/\\\\n/g, '\\n');
            return marked.parse(unescapedText);
        }

        function addMessage(text, classes) {
            const chatBox = document.getElementById('chat-box');
            const div = document.createElement('div');
            div.className = classes + ' text-sm leading-relaxed prose prose-sm max-w-none';
            div.innerHTML = renderMarkdown(text);
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
            return div;
        }

        function sendQuickAction(actionText) {
            if (FAST_RESPONSES[actionText]) {
                addMessage(actionText, "bg-blue-900 text-white p-3 rounded-lg max-w-[85%] self-end ml-auto shadow-sm");
                addMessage(FAST_RESPONSES[actionText], "bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[85%] self-start shadow-sm");
                return;
            }
            document.getElementById('user-input').value = actionText;
            sendMessage();
        }

        async function uploadPDF() {
            const fileInput = document.getElementById('pdf-file');
            const statusDiv = document.getElementById('upload-status');
            if (!fileInput.files.length) return;

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            statusDiv.className = "bg-yellow-100 text-yellow-800 text-xs px-4 py-2 border-b block";
            statusDiv.innerText = "Ingestando '" + fileInput.files[0].name + "'...";

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok) {
                    statusDiv.className = "bg-green-100 text-green-800 text-xs px-4 py-2 border-b block";
                    statusDiv.innerText = data.mensaje;
                } else {
                    statusDiv.className = "bg-red-100 text-red-800 text-xs px-4 py-2 border-b block";
                    statusDiv.innerText = "Error: " + (data.detail || 'Fallo al subir el archivo.');
                }
            } catch (e) {
                statusDiv.className = "bg-red-100 text-red-800 text-xs px-4 py-2 border-b block";
                statusDiv.innerText = "Error de conexión.";
            }
            fileInput.value = '';
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;

            addMessage(text, "bg-blue-900 text-white p-3 rounded-lg max-w-[85%] self-end ml-auto shadow-sm");
            input.value = '';

            const chatBox = document.getElementById('chat-box');
            const botMsgElem = addMessage("Buscando en RIE...", "bg-gray-200 text-gray-600 p-3 rounded-lg max-w-[85%] self-start italic shadow-sm");

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pregunta: text })
                });

                if (!response.ok) {
                    const errData = await response.json();
                    botMsgElem.className = "bg-red-100 text-red-800 p-3 rounded-lg max-w-[85%] self-start shadow-sm not-italic";
                    botMsgElem.innerHTML = renderMarkdown(errData.detail || 'Error en la consulta.');
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let fullText = "";
                let firstChunk = true;
                let renderScheduled = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    if (firstChunk) {
                        botMsgElem.className = "bg-blue-100 text-blue-900 p-3 rounded-lg max-w-[85%] self-start shadow-sm not-italic";
                        firstChunk = false;
                    }

                    fullText += decoder.decode(value, { stream: true });

                    if (!renderScheduled) {
                        renderScheduled = true;
                        requestAnimationFrame(() => {
                            botMsgElem.innerHTML = renderMarkdown(fullText);
                            chatBox.scrollTop = chatBox.scrollHeight;
                            renderScheduled = false;
                        });
                    }
                }
                botMsgElem.innerHTML = renderMarkdown(fullText);

            } catch (e) {
                botMsgElem.className = "bg-red-100 text-red-800 p-3 rounded-lg max-w-[85%] self-start shadow-sm not-italic";
                botMsgElem.innerHTML = "Error de conexión.";
            }
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.post("/chat")
async def chat(request: QueryRequest):
    pregunta_clean = request.pregunta.strip()
    if not pregunta_clean:
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    try:
        pregunta_lower = pregunta_clean.lower()

        # --- RESPUESTAS INMEDIATAS TOLERANTES A ERRORES DE TIPEO (0 ms) ---
        if any(w in pregunta_lower for w in ["matricul", "vacant", "cupo", "postul", "admisi", "inscrip"]):
            async def responder_matricula():
                yield "**📌 Proceso de Admisión y Matrículas - Colegio Nocedal**\n\n" \
                      "• **Sistema Oficial:** Las postulaciones a vacantes y matriculación se realizan exclusivamente por el **Sistema de Admisión Escolar (SAE)** del Mineduc[cite: 49].\n" \
                      "• **Atención Presencial:** Para consultas de listas de espera o documentación, acudir a secretaría de Lunes a Viernes de 08:30 a 17:00 hrs[cite: 35].\n" \
                      "• **Contacto:** admision@nocedal.cl / +56 2 2543 3816."
            return StreamingResponse(responder_matricula(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["hola", "buen", "saludo", "ola", "buenas"]):
            async def responder_saludo():
                yield "¡Hola! Bienvenid@ al Asistente Virtual del Colegio Nocedal. ¿En qué te puedo ayudar hoy? Puedes consultar sobre horarios, uniforme, reglamento, admisión o directores."
            return StreamingResponse(responder_saludo(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["ubica", "ubicu", "direc", "donde", "dónde", "mapa", "llegar"]):
            async def responder_ubicacion():
                yield "**📌 Ubicación y Atención - Colegio Nocedal**\n\n" \
                      "• **Dirección:** Av. El Castillo 1415, La Pintana, Región Metropolitana[cite: 35].\n" \
                      "• **Horario de atención:** Lunes a Viernes de 08:30 a 17:00 hrs[cite: 35]."
            return StreamingResponse(responder_ubicacion(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["direct", "rector"]):
            async def responder_director():
                yield "**📌 Equipo Directivo y Conducto Regular**\n\n" \
                      "• **Director de Ciclo:** Instancia superior para resolver casos pedagógicos o de convivencia no solucionados previamente por el Profesor Jefe o Coordinador[cite: 41].\n" \
                      "• **Conducto Regular:** 1. Profesor de Asignatura ➔ 2. Profesor Jefe ➔ 3. Coordinador de Nivel ➔ 4. Encargado Convivencia Escolar ➔ 5. Director de Ciclo[cite: 41]."
            return StreamingResponse(responder_director(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["telef", "teléf", "contac", "correo", "mail"]):
            async def responder_contacto():
                yield "**📌 Contactos Oficiales - Colegio Nocedal**\n\n* **Teléfono:** +56 2 2543 3816\n* **Correo:** admision@nocedal.cl\n* **Web:** www.nocedal.cl"
            return StreamingResponse(responder_contacto(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["pelo", "corte", "cabello", "barba", "piercing", "presentac"]):
            async def responder_presentacion():
                yield "**📌 Reglamento de Presentación Personal**\n\n" \
                      "• **Corte de Pelo:** Tradicional/escolar sobrio (corto a los lados y atrás), sin teñidos[cite: 28, 72].\n" \
                      "• **Higiene:** Asistir diariamente afeitados (sin barba ni bigote), sin piercings ni accesorios[cite: 28]."
            return StreamingResponse(responder_presentacion(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["unifor", "buzo", "ropa"]):
            async def responder_uniforme():
                yield "**📌 Reglamento de Uniforme Escolar**\n\n" \
                      "Pantalón gris, camisa blanca, corbata institucional, chaleco azul marino con insignia, calcetines azules y zapatos negros[cite: 50]."
            return StreamingResponse(responder_uniforme(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["horari", "salida", "entrada", "hora"]):
            async def responder_horarios():
                yield "**📌 Horarios de Clases**\n\n" \
                      "Ingreso general a las 08:30 hrs (apertura de puertas 07:30 hrs)[cite: 40]. Salida de Básica a las 16:00 hrs[cite: 40]."
            return StreamingResponse(responder_horarios(), media_type="text/plain")

        if any(w in pregunta_lower for w in ["celu", "movil", "móvil", "tablet", "tecno"]):
            async def responder_celular():
                yield "**📌 Uso de Celulares y Tecnología**\n\n" \
                      "Está prohibido el uso de celulares durante las clases sin autorización docente[cite: 29, 61]. En caso de uso no autorizado, se retira por Coordinación hasta el final del día[cite: 29]."
            return StreamingResponse(responder_celular(), media_type="text/plain")

        # --- CONSULTA RAG PARA PREGUNTAS ABIERTAS ---
        count = collection.count()
        if count == 0:
            cargar_todos_los_pdfs()
            count = collection.count()

        if count == 0:
            contexto = "No hay información disponible en el documento."
        else:
            results = await asyncio.to_thread(
                collection.query,
                query_texts=[pregunta_clean],
                n_results=min(2, count)
            )
            context_docs = results.get("documents", [[]])[0]
            contexto = "\n---\n".join(context_docs) if context_docs else "Sin información suficiente."

        prompt = f"""Eres el asistente institucional del Colegio Nocedal. Responde a la pregunta del usuario de forma concisa y educada.

CONTEXTO INSTITUCIONAL DEL RIE:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta_clean}
"""
        gen_config = genai.types.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.1
        )
        model = genai.GenerativeModel(CHAT_MODEL, generation_config=gen_config)

        async def iterar_respuesta():
            try:
                response = await model.generate_content_async(prompt, stream=True)
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text
            except Exception:
                yield "Para información sobre postulaciones y vacantes, por favor comunícate a secretaría (+56 2 2543 3816) o escribe a admision@nocedal.cl."

        return StreamingResponse(iterar_respuesta(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")