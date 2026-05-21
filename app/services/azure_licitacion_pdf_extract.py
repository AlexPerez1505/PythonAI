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
        ar = r.get("analyzeResult", {})
        if ar.get("content"):
            merged_content.append(ar["content"])

        for page in ar.get("pages", []) or []:
            pcopy = dict(page)
            if "pageNumber" in pcopy:
                pcopy["pageNumber"] = page_offset + pcopy["pageNumber"]
            merged_pages.append(pcopy)

        for table in ar.get("tables", []) or []:
            merged_tables.append(table)

        page_offset += len(ar.get("pages", []) or [])

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


# ════════════════════════════════════════════════════════════
# DOCX (texto plano de Word)
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
# OPENAI STRUCTURER (especifico para licitaciones)
# ════════════════════════════════════════════════════════════

LICITACION_SYSTEM_PROMPT = (
    "Eres un asistente que extrae informacion de licitaciones publicas mexicanas. "
    "Responde UNICAMENTE JSON valido, sin markdown."
)


def _build_licitacion_prompt(raw_text: str) -> str:
    compact = (raw_text or "")[:60000]
    return f"""
Analiza el texto de esta licitacion y devuelve UN SOLO JSON con esta estructura exacta:

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
      "pagina": 1
    }}
  ],
  "citas": {{
    "ficha.numero_licitacion": {{"cita": "texto exacto del documento", "fuente": "INV.pdf", "pagina": 1}},
    "ficha.tipo_evento": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "ficha.organismo": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "ficha.objeto_licitacion": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "ficha.medio_participacion": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "fechas_clave.fecha_publicacion": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "fechas_clave.junta_aclaraciones": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "fechas_clave.presentacion_apertura": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "fechas_clave.fallo": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "fechas_clave.vigencia_contrato": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.0": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.1": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.2": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.3": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.4": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.5": {{"cita": "...", "fuente": "...", "pagina": 1}},
    "resumen_ejecutivo.6": {{"cita": "...", "fuente": "...", "pagina": 1}}
  }}
}}

Reglas generales:
- Si un dato no se encuentra, usa exactamente "No se encontro informacion"
- No inventes datos
- Las fechas en formato dd/mm/aaaa cuando sea posible
- En resumen_ejecutivo responde cada pregunta basandote SOLO en el texto
- Las partidas son los items / productos / servicios solicitados
- En "citas" pon el texto LITERAL del documento que respalda cada campo (max 350 caracteres)
- El campo "fuente" debe ser el nombre EXACTO del archivo (los documentos vienen separados por "--- DOCUMENTO: nombre.pdf ---")
- Si no hay cita disponible, omite ese campo de "citas" (no pongas null)

Reglas del checklist:
- Genera ENTRE 20 Y 50 requisitos detectados en el documento
- Cubre todas las categorias: legales, administrativos, tecnicos, anexos requeridos, escritos bajo protesta, garantias, opiniones SAT/IMSS/INFONAVIT, fianzas, fichas tecnicas, etc.
- "formato" usa uno de: "No aplica", "Anexo A", "Anexo B / Plataforma", "Anexo C", "Anexo D", "Formatos de la convocatoria"
- "categoria" usa uno de: "Legal-Administrativo", "Tecnico", "Otro / Tecnico", "Otro"
- "aplicabilidad" usa: "Unico" (un solo entregable) o "Por partida" (uno por cada partida)
- "obligatorio" siempre "Si" (excepto opcionales claramente marcados)
- "cumplimiento" siempre arranca en "-"
- "status" siempre arranca en "Pendiente"
- "prioridad": "Alta" (legal obligatorio), "Media" (tecnico comun), "Baja" (anexos secundarios)
- "fuente" y "pagina" indican de que archivo y pagina se extrajo

Texto de la licitacion:
{compact}
""".strip()


def _get_openai_models() -> List[str]:
    primary = os.getenv("OPENAI_PRIMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
    fallback_raw = os.getenv("OPENAI_FALLBACK_MODELS", "")
    fallbacks = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    seen = set()
    ordered = []
    for m in [primary] + fallbacks:
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)

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
    )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": LICITACION_SYSTEM_PROMPT},
            {"role": "user", "content": _build_licitacion_prompt(raw_text)},
        ],
    }

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
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise Exception(f"Modelo {model} devolvio JSON invalido: {content[:300]}")


def structure_licitacion_with_openai(raw_text: str) -> Dict[str, Any]:
    models = _get_openai_models()

    if not models:
        raise Exception("No hay modelos OpenAI configurados (OPENAI_PRIMARY_MODEL o OPENAI_MODEL)")

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
# PIPELINE PRINCIPAL (CLI entry point)
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
            documents_info.append({"file": name, "status": "error", "error": "Archivo no existe"})
            continue

        ext = p.suffix.lower()

        try:
            _log(f"Procesando {name}...")

            if ext == ".pdf":
                with open(p, "rb") as f:
                    file_bytes = f.read()
                result = analyze_pdf_with_auto_split(file_bytes)
                content = result.get("analyzeResult", {}).get("content", "") or ""

            elif ext in (".docx", ".doc"):
                content = extract_docx_text(str(p))

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
        description="Extrae y estructura licitaciones (Azure + OpenAI)"
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
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()