import hashlib
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

from app.services.progress import write_progress

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _is_size_error(message: str) -> bool:
    text = (message or "").lower()
    return (
        "invalidcontentlength" in text
        or "too large" in text
        or "content length" in text
        or "maximum request size" in text
        or "request entity too large" in text
        or "413" in text
    )


def _analyze_single_pdf_bytes(file_bytes: bytes, model: str = "prebuilt-layout") -> Dict[str, Any]:
    if not AZURE_ENDPOINT:
        raise Exception("Falta AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT en .env")

    if not AZURE_KEY:
        raise Exception("Falta AZURE_DOCUMENT_INTELLIGENCE_KEY en .env")

    file_hash = _sha256(file_bytes)
    file_size = len(file_bytes)

    _log(f"[Azure] Enviando PDF a analizar")
    _log(f"[Azure] Modelo: {model}")
    _log(f"[Azure] API version: {AZURE_API_VERSION}")
    _log(f"[Azure] PDF size bytes: {file_size}")
    _log(f"[Azure] PDF sha256: {file_hash}")

    write_progress(15, "Enviando PDF a Azure", f"{file_size // 1024} KB")

    url = f"{AZURE_ENDPOINT}/documentintelligence/documentModels/{model}:analyze?api-version={AZURE_API_VERSION}"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/pdf",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    response = requests.post(url, headers=headers, data=file_bytes, timeout=180)

    if not response.ok:
        raise Exception(f"Error iniciando Azure analysis: {response.status_code} - {response.text}")

    operation_location = response.headers.get("operation-location")

    if not operation_location:
        raise Exception("Azure no devolvió operation-location")

    _log(f"[Azure] Operation location recibido: {operation_location}")

    while True:
        poll = requests.get(
            operation_location,
            headers={
                "Ocp-Apim-Subscription-Key": AZURE_KEY,
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
            },
            timeout=180,
        )

        if not poll.ok:
            raise Exception(f"Error consultando Azure result: {poll.status_code} - {poll.text}")

        data = poll.json()
        status = data.get("status")

        _log(f"[Azure] Estado actual: {status}")
        write_progress(25, "Azure está leyendo el documento", f"estado: {status}")

        if status in ["notStarted", "running"]:
            time.sleep(2)
            continue

        if status != "succeeded":
            raise Exception(f"Azure terminó con estado no exitoso: {data}")

        analyze_result = data.get("analyzeResult", {}) or {}
        pages = analyze_result.get("pages", []) or []
        tables = analyze_result.get("tables", []) or []
        content = analyze_result.get("content", "") or ""

        _log("[Azure] Análisis terminado correctamente")
        _log(f"[Azure] Páginas detectadas: {len(pages)}")
        _log(f"[Azure] Tablas detectadas: {len(tables)}")
        _log(f"[Azure] Caracteres extraídos: {len(content)}")

        write_progress(40, "Azure terminó la lectura", f"{len(tables)} tablas, {len(pages)} páginas")

        return data


def split_pdf_bytes(file_bytes: bytes, pages_per_chunk: int) -> List[bytes]:
    if pages_per_chunk < 1:
        raise Exception("pages_per_chunk debe ser mayor o igual a 1")

    reader = PdfReader(BytesIO(file_bytes))
    total_pages = len(reader.pages)

    _log(f"[Split] Total páginas PDF original: {total_pages}")
    _log(f"[Split] Páginas por bloque: {pages_per_chunk}")

    chunks = []

    for start in range(0, total_pages, pages_per_chunk):
        end = min(start + pages_per_chunk, total_pages)

        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])

        output = BytesIO()
        writer.write(output)

        chunk_bytes = output.getvalue()
        chunks.append(chunk_bytes)

        _log(f"[Split] Chunk creado páginas {start + 1}-{end} | bytes={len(chunk_bytes)} | sha256={_sha256(chunk_bytes)}")

    return chunks


def _offset_spans(spans: List[Dict[str, Any]], content_offset: int) -> List[Dict[str, Any]]:
    new_spans = []
    for span in spans or []:
        span_copy = dict(span)
        if "offset" in span_copy and isinstance(span_copy["offset"], int):
            span_copy["offset"] = content_offset + span_copy["offset"]
        new_spans.append(span_copy)
    return new_spans


def _offset_page_number_in_regions(regions: List[Dict[str, Any]], page_offset: int) -> List[Dict[str, Any]]:
    new_regions = []
    for region in regions or []:
        region_copy = dict(region)
        if "pageNumber" in region_copy and isinstance(region_copy["pageNumber"], int):
            region_copy["pageNumber"] = page_offset + region_copy["pageNumber"]
        new_regions.append(region_copy)
    return new_regions


def merge_azure_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_content_parts = []
    merged_pages = []
    merged_tables = []

    current_page_offset = 0
    current_content_offset = 0

    for result_index, result in enumerate(results, start=1):
        analyze_result = result.get("analyzeResult", {}) or {}

        content = analyze_result.get("content", "") or ""
        pages = analyze_result.get("pages", []) or []
        tables = analyze_result.get("tables", []) or []

        _log(
            f"[Merge] Resultado {result_index}: "
            f"pages={len(pages)} tables={len(tables)} content_chars={len(content)} "
            f"page_offset={current_page_offset} content_offset={current_content_offset}"
        )

        if content:
            merged_content_parts.append(content)

        for page in pages:
            page_copy = dict(page)

            if "pageNumber" in page_copy and isinstance(page_copy["pageNumber"], int):
                page_copy["pageNumber"] = current_page_offset + page_copy["pageNumber"]

            if "spans" in page_copy:
                page_copy["spans"] = _offset_spans(page_copy.get("spans", []), current_content_offset)

            if "words" in page_copy:
                new_words = []
                for word in page_copy.get("words", []) or []:
                    word_copy = dict(word)
                    if "span" in word_copy and isinstance(word_copy["span"], dict):
                        word_copy["span"] = _offset_spans([word_copy["span"]], current_content_offset)[0]
                    new_words.append(word_copy)
                page_copy["words"] = new_words

            if "lines" in page_copy:
                new_lines = []
                for line in page_copy.get("lines", []) or []:
                    line_copy = dict(line)
                    if "spans" in line_copy:
                        line_copy["spans"] = _offset_spans(line_copy.get("spans", []), current_content_offset)
                    new_lines.append(line_copy)
                page_copy["lines"] = new_lines

            merged_pages.append(page_copy)

        for table in tables:
            table_copy = dict(table)

            if "boundingRegions" in table_copy:
                table_copy["boundingRegions"] = _offset_page_number_in_regions(
                    table_copy.get("boundingRegions", []),
                    current_page_offset,
                )

            if "spans" in table_copy:
                table_copy["spans"] = _offset_spans(table_copy.get("spans", []), current_content_offset)

            new_cells = []
            for cell in table_copy.get("cells", []) or []:
                cell_copy = dict(cell)

                if "boundingRegions" in cell_copy:
                    cell_copy["boundingRegions"] = _offset_page_number_in_regions(
                        cell_copy.get("boundingRegions", []),
                        current_page_offset,
                    )

                if "spans" in cell_copy:
                    cell_copy["spans"] = _offset_spans(cell_copy.get("spans", []), current_content_offset)

                new_cells.append(cell_copy)

            table_copy["cells"] = new_cells
            merged_tables.append(table_copy)

        current_page_offset += len(pages)

        if content:
            current_content_offset += len(content) + 2

    merged_content = "\n\n".join(merged_content_parts)

    _log("[Merge] Merge terminado")
    _log(f"[Merge] Total páginas: {len(merged_pages)}")
    _log(f"[Merge] Total tablas: {len(merged_tables)}")
    _log(f"[Merge] Total caracteres: {len(merged_content)}")

    write_progress(40, "Azure terminó la lectura", f"{len(merged_tables)} tablas, {len(merged_pages)} páginas")

    return {
        "status": "succeeded",
        "analyzeResult": {
            "content": merged_content,
            "pages": merged_pages,
            "tables": merged_tables,
        },
    }


def analyze_pdf_with_auto_split(file_bytes: bytes, model: str = "prebuilt-layout", pages_per_chunk: int = 5) -> Dict[str, Any]:
    if not file_bytes:
        raise Exception("El PDF está vacío o no se recibieron bytes.")

    original_hash = _sha256(file_bytes)
    original_size = len(file_bytes)

    _log("========== NUEVO ANALISIS PDF ==========")
    _log(f"[PDF Original] size bytes: {original_size}")
    _log(f"[PDF Original] sha256: {original_hash}")
    _log(f"[PDF Original] model: {model}")
    _log(f"[PDF Original] pages_per_chunk inicial: {pages_per_chunk}")

    try:
        _log("[Azure] Intentando analizar PDF completo...")
        result = _analyze_single_pdf_bytes(file_bytes, model=model)
        _log("[Azure] PDF completo analizado correctamente.")
        return result

    except Exception as e:
        error_message = str(e)
        _log(f"[Azure] Error analizando PDF completo: {error_message}")
        if not _is_size_error(error_message):
            raise
        _log("[Azure] Azure rechazó el PDF completo por tamaño. Se intentará dividir.")

    current_chunk_size = pages_per_chunk

    while current_chunk_size >= 1:
        _log(f"[Split] Intentando dividir PDF en bloques de {current_chunk_size} página(s)...")
        write_progress(20, "Dividiendo el PDF", f"bloques de {current_chunk_size} página(s)")

        chunks = split_pdf_bytes(file_bytes, pages_per_chunk=current_chunk_size)

        partial_results = []
        failed_due_to_size = False

        for index, chunk_bytes in enumerate(chunks, start=1):
            try:
                _log(f"[Azure] Procesando chunk {index}/{len(chunks)} con máximo {current_chunk_size} página(s)...")
                _log(f"[Azure] Chunk {index} size bytes: {len(chunk_bytes)}")
                _log(f"[Azure] Chunk {index} sha256: {_sha256(chunk_bytes)}")

                write_progress(20 + int(20 * index / max(len(chunks), 1)), "Azure leyendo el PDF", f"bloque {index} de {len(chunks)}")

                partial_result = _analyze_single_pdf_bytes(chunk_bytes, model=model)
                partial_results.append(partial_result)

                _log(f"[Azure] Chunk {index}/{len(chunks)} procesado correctamente.")

            except Exception as e:
                error_message = str(e)
                _log(f"[Azure] Error procesando chunk {index}/{len(chunks)}: {error_message}")

                if _is_size_error(error_message) and current_chunk_size > 1:
                    _log("[Azure] Chunk todavía muy grande. Se reducirá el tamaño del bloque.")
                    failed_due_to_size = True
                    break

                raise

        if not failed_due_to_size:
            _log("[Azure] Todos los chunks procesados correctamente. Haciendo merge...")
            return merge_azure_results(partial_results)

        current_chunk_size = current_chunk_size // 2

    raise Exception("No se pudo procesar el PDF: incluso dividido en bloques mínimos Azure lo rechazó por tamaño.")