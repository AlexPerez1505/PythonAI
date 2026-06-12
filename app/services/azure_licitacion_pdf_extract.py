import argparse
import json
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _is_size_error(message: str) -> bool:
    text = (message or "").lower()
    return "invalidcontentlength" in text or "too large" in text


def _norm(v: Any) -> Optional[str]:
    if v is None:
        return None

    t = re.sub(r"\s+", " ", str(v)).strip()
    return t or None


def _is_empty_value(value: Any) -> bool:
    text = _norm(value) or ""
    low = text.lower()

    return (
        not text
        or low in [
            "no se encontro informacion",
            "no se encontró información",
            "no se encontro información",
            "no se encontró informacion",
            "no aplica",
            "n/a",
            "na",
            "-",
            "sin informacion",
            "sin información",
        ]
    )


def _clean_for_search(text: Any) -> str:
    text = _norm(text) or ""
    text = text.lower()
    text = re.sub(r"[^\wáéíóúñü\s]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ════════════════════════════════════════════════════════════
# AZURE DOCUMENT INTELLIGENCE
# ════════════════════════════════════════════════════════════

def _analyze_single_pdf_bytes(file_bytes: bytes, model: str = "prebuilt-layout") -> Dict[str, Any]:
    if not AZURE_ENDPOINT:
        raise Exception("Falta AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT en .env")

    if not AZURE_KEY:
        raise Exception("Falta AZURE_DOCUMENT_INTELLIGENCE_KEY en .env")

    url = f"{AZURE_ENDPOINT}/documentintelligence/documentModels/{model}:analyze?api-version={AZURE_API_VERSION}"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/pdf",
    }

    response = requests.post(url, headers=headers, data=file_bytes, timeout=180)

    if not response.ok:
        raise Exception(f"Azure analyze error: {response.status_code} - {response.text}")

    operation_location = response.headers.get("operation-location")
    if not operation_location:
        raise Exception("Azure no devolvio operation-location")

    while True:
        poll = requests.get(
            operation_location,
            headers={"Ocp-Apim-Subscription-Key": AZURE_KEY},
            timeout=180,
        )

        if not poll.ok:
            raise Exception(f"Azure poll error: {poll.status_code} - {poll.text}")

        data = poll.json()
        status = data.get("status")

        if status in ["notStarted", "running"]:
            time.sleep(2)
            continue

        if status != "succeeded":
            raise Exception(f"Azure status no exitoso: {data}")

        return data


def _split_pdf_bytes(file_bytes: bytes, pages_per_chunk: int) -> List[bytes]:
    reader = PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)

    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)
        writer = PdfWriter()

        for i in range(start, end):
            writer.add_page(reader.pages[i])

        output = BytesIO()
        writer.write(output)
        chunks.append(output.getvalue())

    return chunks


def _merge_azure_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_content = []
    merged_pages = []
    merged_tables = []
    page_offset = 0

    for r in results:
        ar = r.get("analyzeResult", {}) or {}

        if ar.get("content"):
            merged_content.append(ar["content"])

        pages = ar.get("pages", []) or []

        for page in pages:
            pcopy = dict(page)

            if "pageNumber" in pcopy:
                pcopy["pageNumber"] = page_offset + pcopy["pageNumber"]

            merged_pages.append(pcopy)

        for table in ar.get("tables", []) or []:
            merged_tables.append(table)

        page_offset += len(pages)

    return {
        "status": "succeeded",
        "analyzeResult": {
            "content": "\n\n".join(merged_content),
            "pages": merged_pages,
            "tables": merged_tables,
        },
    }


def analyze_pdf_with_auto_split(file_bytes: bytes, pages_per_chunk: int = 5) -> Dict[str, Any]:
    try:
        _log("Azure: intentando PDF completo...")
        return _analyze_single_pdf_bytes(file_bytes)
    except Exception as e:
        if not _is_size_error(str(e)):
            raise

        _log("PDF rechazado por tamano, dividiendo...")

    chunk_size = pages_per_chunk

    while chunk_size >= 1:
        _log(f"Probando con chunks de {chunk_size} paginas...")

        chunks = _split_pdf_bytes(file_bytes, chunk_size)
        partials = []
        failed_size = False

        for idx, ch_bytes in enumerate(chunks, start=1):
            try:
                _log(f"Procesando chunk {idx}/{len(chunks)}...")
                partials.append(_analyze_single_pdf_bytes(ch_bytes))
            except Exception as e:
                if _is_size_error(str(e)) and chunk_size > 1:
                    _log(f"Chunk {idx} aun grande, reduciendo...")
                    failed_size = True
                    break

                raise

        if not failed_size:
            return _merge_azure_results(partials)

        chunk_size = chunk_size // 2

    raise Exception("Azure rechazo el PDF incluso dividido al minimo.")


def azure_result_to_page_marked_text(analyze_result: Dict[str, Any]) -> str:
    """
    Convierte el resultado de Azure en texto con marcas de página:
    [PAGINA 1]
    texto...

    Esto permite que OpenAI y el fallback local sepan de qué página salió cada dato.
    """
    pages = analyze_result.get("pages", []) or []
    page_blocks = []

    for page in pages:
        page_number = page.get("pageNumber")
        lines = page.get("lines", []) or []

        line_texts = []

        for line in lines:
            content = _norm(line.get("content"))
            if content:
                line_texts.append(content)

        if line_texts:
            page_blocks.append(
                f"[PAGINA {page_number}]\n" + "\n".join(line_texts)
            )

    if page_blocks:
        return "\n\n".join(page_blocks)

    return analyze_result.get("content", "") or ""


# ════════════════════════════════════════════════════════════
# DOCX
# ════════════════════════════════════════════════════════════

def extract_docx_text(file_path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise Exception("Falta paquete 'python-docx'. Instala: pip install python-docx")

    doc = Document(file_path)
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


# ════════════════════════════════════════════════════════════
# EVIDENCE / CITAS FALLBACK
# ════════════════════════════════════════════════════════════

def _extract_source_and_page(raw_text: str, pos: int) -> Dict[str, Any]:
    before = raw_text[:pos]

    doc_matches = list(re.finditer(r"--- DOCUMENTO:\s*(.*?)\s*---", before))
    page_matches = list(re.finditer(r"\[PAGINA\s+(\d+)\]", before, flags=re.IGNORECASE))

    fuente = doc_matches[-1].group(1).strip() if doc_matches else ""
    pagina = int(page_matches[-1].group(1)) if page_matches else None

    return {
        "fuente": fuente,
        "pagina": pagina,
    }


def _make_quote(raw_text: str, pos: int, length: int = 350) -> str:
    start = max(0, pos - 100)
    end = min(len(raw_text), pos + length)

    quote = raw_text[start:end]
    quote = re.sub(r"--- DOCUMENTO:\s*.*?\s*---", "", quote)
    quote = re.sub(r"\[PAGINA\s+\d+\]", "", quote, flags=re.IGNORECASE)
    quote = re.sub(r"\s+", " ", quote).strip()

    return quote[:350]


def _find_evidence_for_value(raw_text: str, value: Any) -> Optional[Dict[str, Any]]:
    """
    Busca el valor extraído dentro del texto original.
    Si lo encuentra, arma:
    {
      "cita": "...",
      "fuente": "archivo.pdf",
      "pagina": 1
    }

    Esto corrige casos donde OpenAI llenó ficha/resumen pero omitió citas.
    """
    value_text = _norm(value)

    if not value_text or _is_empty_value(value_text):
        return None

    raw_low = raw_text.lower()
    value_low = value_text.lower()

    pos = raw_low.find(value_low)

    if pos < 0:
        words = [
            w for w in re.findall(r"[\wáéíóúñü]+", value_low, flags=re.IGNORECASE)
            if len(w) >= 4
        ]

        # Buscar por frases parciales fuertes.
        # Ejemplo: "invitacion cuando menos tres personas".
        for size in range(min(8, len(words)), 2, -1):
            for i in range(0, len(words) - size + 1):
                phrase_words = words[i:i + size]
                pattern = r"\b" + r"\W+".join(map(re.escape, phrase_words)) + r"\b"
                match = re.search(pattern, raw_low, flags=re.IGNORECASE)

                if match:
                    pos = match.start()
                    break

            if pos >= 0:
                break

    if pos < 0:
        return None

    meta = _extract_source_and_page(raw_text, pos)

    return {
        "cita": _make_quote(raw_text, pos),
        "fuente": meta["fuente"],
        "pagina": meta["pagina"],
    }


def ensure_ficha_resumen_citations(structured: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    Garantiza citas para:
    - ficha.*
    - fechas_clave.*
    - resumen_ejecutivo.N

    Si OpenAI ya devolvió citas válidas, las respeta.
    Si faltan, intenta reconstruirlas buscando el valor en el texto original.
    """
    if not isinstance(structured, dict):
        return structured

    citas = structured.get("citas")

    if not isinstance(citas, dict):
        citas = {}

    ficha = structured.get("ficha") or {}
    fechas = structured.get("fechas_clave") or {}
    resumen = structured.get("resumen_ejecutivo") or []

    if isinstance(ficha, dict):
        for field, value in ficha.items():
            key = f"ficha.{field}"

            if key not in citas and not _is_empty_value(value):
                evidence = _find_evidence_for_value(raw_text, value)

                if evidence:
                    citas[key] = evidence

    if isinstance(fechas, dict):
        for field, value in fechas.items():
            key = f"fechas_clave.{field}"

            if key not in citas and not _is_empty_value(value):
                evidence = _find_evidence_for_value(raw_text, value)

                if evidence:
                    citas[key] = evidence

    if isinstance(resumen, list):
        for idx, item in enumerate(resumen):
            if not isinstance(item, dict):
                continue

            key = f"resumen_ejecutivo.{idx}"
            answer = item.get("respuesta")

            if key not in citas and not _is_empty_value(answer):
                evidence = _find_evidence_for_value(raw_text, answer)

                if evidence:
                    citas[key] = evidence

    structured["citas"] = citas
    return structured


def ensure_checklist_citations(structured: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    """
    También mejora checklist_sugerido:
    - Si un requisito tiene fuente/página vacío, intenta encontrar evidencia.
    - Agrega campo cita si puede encontrarla.
    """
    if not isinstance(structured, dict):
        return structured

    checklist = structured.get("checklist_sugerido")

    if not isinstance(checklist, list):
        return structured

    for item in checklist:
        if not isinstance(item, dict):
            continue

        if item.get("fuente") and item.get("pagina") and item.get("cita"):
            continue

        search_value = (
            item.get("requisito")
            or item.get("descripcion")
            or item.get("formato")
        )

        evidence = _find_evidence_for_value(raw_text, search_value)

        if evidence:
            item.setdefault("cita", evidence.get("cita"))
            item.setdefault("fuente", evidence.get("fuente"))
            item.setdefault("pagina", evidence.get("pagina"))

    return structured


def postprocess_structured_data(structured: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    structured = ensure_ficha_resumen_citations(structured, raw_text)
    structured = ensure_checklist_citations(structured, raw_text)
    return structured


# ════════════════════════════════════════════════════════════
# OPENAI STRUCTURER
# ════════════════════════════════════════════════════════════

LICITACION_SYSTEM_PROMPT = (
    "Eres un asistente experto en extraccion de informacion de licitaciones publicas mexicanas. "
    "Debes responder UNICAMENTE JSON valido, sin markdown, sin explicaciones y sin texto adicional."
)


def _build_licitacion_prompt(raw_text: str) -> str:
    compact = (raw_text or "")[:90000]

    return f"""
Analiza el texto de esta licitacion y devuelve UN SOLO JSON valido con esta estructura exacta:

{{
  "ficha": {{
    "numero_licitacion": "...",
    "tipo_evento": "...",
    "organismo": "...",
    "objeto_licitacion": "...",
    "medio_participacion": "..."
  }},
  "fechas_clave": {{
    "fecha_publicacion": "...",
    "junta_aclaraciones": "...",
    "presentacion_apertura": "...",
    "fallo": "...",
    "vigencia_contrato": "..."
  }},
  "resumen_ejecutivo": [
    {{"pregunta": "Cuanto tiempo tengo para implementar?", "respuesta": "..."}},
    {{"pregunta": "Es necesario demostrar experiencia previa o acreditar experiencia?", "respuesta": "..."}},
    {{"pregunta": "Se mencionan penas convencionales, multas, deducciones u otras sanciones en caso de incumplimiento?", "respuesta": "..."}},
    {{"pregunta": "Cual es el periodo de garantia a ofertar?", "respuesta": "..."}},
    {{"pregunta": "Cual es el sistema de evaluacion?", "respuesta": "..."}},
    {{"pregunta": "Se requieren cartas de apoyo?", "respuesta": "..."}},
    {{"pregunta": "Se deben entregar muestras fisicas?", "respuesta": "..."}}
  ],
  "partidas": [
    {{"numero": 1, "descripcion": "...", "unidad": "...", "cantidad": 0}}
  ],
  "checklist_sugerido": [
    {{
      "requisito": "Nombre del requisito a presentar",
      "descripcion": "Detalle de que debe contener el documento",
      "formato": "No aplica",
      "categoria": "Legal-Administrativo",
      "aplicabilidad": "Unico",
      "obligatorio": "Si",
      "cumplimiento": "-",
      "status": "Pendiente",
      "prioridad": "Media",
      "fuente": "INV.pdf",
      "pagina": 1,
      "cita": "texto literal del documento que respalda el requisito"
    }}
  ],
  "citas": {{
    "ficha.numero_licitacion": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "ficha.tipo_evento": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "ficha.organismo": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "ficha.objeto_licitacion": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "ficha.medio_participacion": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},

    "fechas_clave.fecha_publicacion": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "fechas_clave.junta_aclaraciones": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "fechas_clave.presentacion_apertura": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "fechas_clave.fallo": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "fechas_clave.vigencia_contrato": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},

    "resumen_ejecutivo.0": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.1": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.2": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.3": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.4": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.5": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "resumen_ejecutivo.6": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}}
  }}
}}

Reglas obligatorias:
- Responde SOLO JSON valido.
- No uses markdown.
- No inventes datos.
- Si un dato no se encuentra, usa exactamente "No se encontro informacion".
- Las fechas deben ir en formato dd/mm/aaaa cuando sea posible.
- En resumen_ejecutivo responde cada pregunta basandote SOLO en el texto.
- Las partidas son los items, productos o servicios solicitados.
- En "citas" pon texto LITERAL del documento que respalda cada campo.
- Cada cita debe tener maximo 350 caracteres.
- El campo "fuente" debe ser el nombre EXACTO del archivo que aparece en el encabezado "--- DOCUMENTO: nombre.pdf ---".
- El campo "pagina" debe tomarse de la marca [PAGINA X] mas cercana antes del texto citado.
- Para cada campo de "ficha" que NO sea "No se encontro informacion", es OBLIGATORIO crear su entrada correspondiente en "citas".
- Para cada campo de "fechas_clave" que NO sea "No se encontro informacion", es OBLIGATORIO crear su entrada correspondiente en "citas".
- Para cada respuesta de "resumen_ejecutivo" que NO sea "No se encontro informacion", es OBLIGATORIO crear su entrada correspondiente en "citas".
- Solo omite una entrada de "citas" cuando el valor sea exactamente "No se encontro informacion".
- La cita debe ser un fragmento literal del documento, no una explicacion.
- Si el dato aparece en varias paginas, usa la cita mas directa y especifica.

Reglas del checklist:
- Genera ENTRE 20 Y 50 requisitos detectados en el documento.
- Cubre todas las categorias: legales, administrativos, tecnicos, anexos requeridos, escritos bajo protesta, garantias, opiniones SAT/IMSS/INFONAVIT, fianzas, fichas tecnicas, etc.
- "formato" usa uno de: "No aplica", "Anexo A", "Anexo B / Plataforma", "Anexo C", "Anexo D", "Formatos de la convocatoria".
- "categoria" usa uno de: "Legal-Administrativo", "Tecnico", "Otro / Tecnico", "Otro".
- "aplicabilidad" usa: "Unico" o "Por partida".
- "obligatorio" siempre "Si", excepto opcionales claramente marcados.
- "cumplimiento" siempre arranca en "-".
- "status" siempre arranca en "Pendiente".
- "prioridad": "Alta" para legal obligatorio, "Media" para tecnico comun, "Baja" para anexos secundarios.
- Cada item de checklist_sugerido debe incluir "fuente", "pagina" y "cita".
- "cita" en checklist_sugerido debe ser texto literal del documento.

Texto de la licitacion:
{compact}
""".strip()


def _get_openai_models() -> List[str]:
    """
    Modelo principal solicitado: gpt-5.4.

    Si tu cuenta/API no reconoce gpt-5.4, el script prueba fallbacks.
    También puedes controlar esto desde .env:

    OPENAI_PRIMARY_MODEL=gpt-5.4
    OPENAI_FALLBACK_MODELS=gpt-5.5,gpt-5,gpt-4.1,gpt-4o
    """
    primary = os.getenv("OPENAI_PRIMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4"))
    fallback_raw = os.getenv("OPENAI_FALLBACK_MODELS", "gpt-5.5,gpt-5,gpt-4.1,gpt-4o")
    fallbacks = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    seen = set()
    ordered = []

    for model in [primary] + fallbacks:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)

    return ordered


def _is_model_error(msg: str) -> bool:
    msg = (msg or "").lower()

    return any(s in msg for s in [
        "model_not_found",
        "does not have access",
        "invalid_request_error",
        "not supported",
        "unsupported",
        "model `",
        "the model",
        "does not exist",
    ])


def _call_openai_once(model: str, raw_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise Exception("Falta OPENAI_API_KEY en .env")

    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("Falta paquete 'openai'. Instala: pip install openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    is_reasoning = (
        model.startswith("gpt-5")
        or model.startswith("o1")
        or model.startswith("o3")
        or model.startswith("o4")
    )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": LICITACION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_licitacion_prompt(raw_text)},
        ],
    }

    # Los modelos reasoning pueden no aceptar temperature/response_format
    # dependiendo de la version del SDK/modelo.
    if not is_reasoning:
        kwargs["temperature"] = 0.1
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or "{}"

    clean = content.strip()

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise Exception(f"Modelo {model} devolvio JSON invalido: {content[:500]}")


def structure_licitacion_with_openai(raw_text: str) -> Dict[str, Any]:
    models = _get_openai_models()

    if not models:
        raise Exception("No hay modelos OpenAI configurados")

    last_error: Optional[Exception] = None

    for model in models:
        try:
            _log(f"OpenAI: probando modelo {model}")
            return _call_openai_once(model, raw_text)
        except Exception as e:
            msg = str(e)
            _log(f"OpenAI fallo con modelo {model}: {msg}")
            last_error = e

            if not _is_model_error(msg):
                raise

    raise last_error or Exception("Todos los modelos OpenAI fallaron")


# ════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════

def process_files(file_paths: List[str], include_raw: bool = False) -> Dict[str, Any]:
    if not file_paths:
        raise Exception("No se recibieron archivos")

    combined_parts = []
    documents_info = []

    for fp in file_paths:
        p = Path(fp)
        name = p.name

        if not p.exists():
            documents_info.append({
                "file": name,
                "status": "error",
                "error": "Archivo no existe",
            })
            continue

        ext = p.suffix.lower()

        try:
            _log(f"Procesando {name}...")

            if ext == ".pdf":
                with open(p, "rb") as f:
                    file_bytes = f.read()

                result = analyze_pdf_with_auto_split(file_bytes)
                analyze_result = result.get("analyzeResult", {}) or {}
                content = azure_result_to_page_marked_text(analyze_result)

            elif ext in (".docx", ".doc"):
                content = "[PAGINA desconocida]\n" + extract_docx_text(str(p))

            else:
                documents_info.append({
                    "file": name,
                    "status": "error",
                    "error": f"Extension no soportada: {ext}",
                })
                continue

            if content:
                combined_parts.append(f"--- DOCUMENTO: {name} ---\n{content}")
                documents_info.append({
                    "file": name,
                    "status": "ok",
                    "text_length": len(content),
                })
            else:
                documents_info.append({
                    "file": name,
                    "status": "empty",
                    "error": "Sin contenido extraido",
                })

        except Exception as e:
            _log(f"Error con {name}: {e}")
            documents_info.append({
                "file": name,
                "status": "error",
                "error": str(e),
            })

    combined_text = "\n\n".join(combined_parts)

    if not combined_text:
        return {
            "ok": False,
            "error": "No se pudo extraer texto de ningun archivo",
            "documents": documents_info,
        }

    try:
        _log("Estructurando con OpenAI...")
        structured = structure_licitacion_with_openai(combined_text)

        _log("Postprocesando citas, fuentes y paginas...")
        structured = postprocess_structured_data(structured, combined_text)

    except Exception as e:
        return {
            "ok": False,
            "error": f"Error OpenAI: {e}",
            "documents": documents_info,
        }

    output = {
        "ok": True,
        "structured": structured,
        "documents": documents_info,
    }

    if include_raw:
        output["raw_text_combined"] = combined_text

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extrae y estructura licitaciones con Azure Document Intelligence + OpenAI"
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Rutas absolutas a archivos PDF/DOCX",
    )

    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Incluir texto crudo combinado en la salida",
    )

    args = parser.parse_args()

    try:
        result = process_files(args.files, include_raw=args.include_raw)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0 if result.get("ok") else 1)

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e),
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()