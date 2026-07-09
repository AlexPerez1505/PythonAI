import hashlib
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

from app.services.progress import write_progress

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")

# Paralelización de bloques (tuneable por .env)
AZURE_MAX_WORKERS = int(os.getenv("AZURE_MAX_WORKERS", "5"))
AZURE_PAGES_PER_CHUNK = int(os.getenv("AZURE_PAGES_PER_CHUNK", "20"))
# Si el PDF tiene MÁS páginas que esto, se parte en paralelo desde el inicio.
AZURE_FORCE_SPLIT_PAGES = int(os.getenv("AZURE_FORCE_SPLIT_PAGES", "35"))
# Caché por hash: si un PDF ya se analizó antes, se reutiliza y NO se vuelve a llamar a Azure.
AZURE_CACHE_ENABLED = os.getenv("AZURE_CACHE_ENABLED", "1") == "1"
AZURE_CACHE_DIR = BASE_DIR / "storage" / "azure_cache"
AZURE_POLL_SECONDS = float(os.getenv("AZURE_POLL_SECONDS", "1.5"))


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


# ---------------------------------------------------------------------------
# CACHÉ POR HASH
# ---------------------------------------------------------------------------

def _cache_path(file_hash: str, model: str) -> Path:
    return AZURE_CACHE_DIR / f"{model}_{file_hash}.json"


def _load_from_cache(file_hash: str, model: str) -> Optional[Dict[str, Any]]:
    if not AZURE_CACHE_ENABLED:
        return None
    path = _cache_path(file_hash, model)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("analyzeResult"):
            _log(f"[Cache] HIT: usando resultado guardado ({path.name}). Se salta Azure.")
            return data
    except Exception as e:
        _log(f"[Cache] No se pudo leer caché ({e}). Se reanaliza.")
    return None


def _save_to_cache(file_hash: str, model: str, result: Dict[str, Any]) -> None:
    if not AZURE_CACHE_ENABLED:
        return
    try:
        AZURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(file_hash, model)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        _log(f"[Cache] Guardado resultado para reintentos futuros ({path.name}).")
    except Exception as e:
        _log(f"[Cache] No se pudo guardar caché ({e}).")


# ---------------------------------------------------------------------------
# AZURE
# ---------------------------------------------------------------------------

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

        if status in ["notStarted", "running"]:
            time.sleep(AZURE_POLL_SECONDS)
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

        _log(f"[Split] Chunk creado páginas {start + 1}-{end} | bytes={len(chunk_bytes)}")

    return chunks


def _count_pdf_pages(file_bytes: bytes) -> int:
    try:
        return len(PdfReader(BytesIO(file_bytes)).pages)
    except Exception:
        return 0


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


def _analyze_chunks_parallel(chunks: List[bytes], model: str, max_workers: int):
    """Procesa los bloques EN PARALELO pero devuelve los resultados EN ORDEN."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
    errors: Dict[int, Exception] = {}
    workers = max(1, min(max_workers, len(chunks)))
    total = len(chunks)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_analyze_single_pdf_bytes, ch, model): idx
            for idx, ch in enumerate(chunks)
        }
        done = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
                _log(f"[Azure] Chunk {idx + 1}/{total} procesado correctamente.")
            except Exception as e:
                errors[idx] = e
                _log(f"[Azure] Error procesando chunk {idx + 1}/{total}: {e}")
            done += 1
            # Progreso REAL: sube de 15% a 40% conforme caen los bloques.
            write_progress(15 + int(25 * done / max(total, 1)), "Azure leyendo el PDF", f"bloque {done} de {total}")

    return results, errors


def _split_and_analyze_parallel(file_bytes: bytes, model: str, pages_per_chunk: int) -> Dict[str, Any]:
    current_chunk_size = pages_per_chunk

    while current_chunk_size >= 1:
        _log(f"[Split] Dividiendo en bloques de {current_chunk_size} pág. (paralelo, {AZURE_MAX_WORKERS} workers)...")
        write_progress(15, "Dividiendo el PDF", f"bloques de {current_chunk_size} página(s)")

        chunks = split_pdf_bytes(file_bytes, pages_per_chunk=current_chunk_size)

        results, errors = _analyze_chunks_parallel(chunks, model, AZURE_MAX_WORKERS)

        other_error = next((e for e in errors.values() if not _is_size_error(str(e))), None)
        if other_error is not None:
            raise other_error

        if any(_is_size_error(str(e)) for e in errors.values()):
            if current_chunk_size > 1:
                _log("[Azure] Algún chunk todavía muy grande. Se reduce el tamaño del bloque.")
                current_chunk_size = current_chunk_size // 2
                continue
            raise Exception("No se pudo procesar el PDF: incluso dividido en bloques mínimos Azure lo rechazó.")

        _log("[Azure] Todos los chunks procesados. Haciendo merge...")
        return merge_azure_results(results)

    raise Exception("No se pudo procesar el PDF: incluso dividido en bloques mínimos Azure lo rechazó.")


def analyze_pdf_with_auto_split(file_bytes: bytes, model: str = "prebuilt-layout", pages_per_chunk: int = None) -> Dict[str, Any]:
    if not file_bytes:
        raise Exception("El PDF está vacío o no se recibieron bytes.")

    if pages_per_chunk is None:
        pages_per_chunk = AZURE_PAGES_PER_CHUNK

    original_hash = _sha256(file_bytes)
    original_size = len(file_bytes)
    total_pages = _count_pdf_pages(file_bytes)

    _log("========== NUEVO ANALISIS PDF ==========")
    _log(f"[PDF Original] size bytes: {original_size}")
    _log(f"[PDF Original] sha256: {original_hash}")
    _log(f"[PDF Original] model: {model}")
    _log(f"[PDF Original] total páginas: {total_pages}")
    _log(f"[PDF Original] pages_per_chunk: {pages_per_chunk}")

    write_progress(12, "Preparando análisis", f"{total_pages} páginas")

    # 1) CACHÉ: si este PDF ya se analizó antes, lo reutilizamos y saltamos Azure.
    cached = _load_from_cache(original_hash, model)
    if cached is not None:
        write_progress(40, "Documento en caché", "Reutilizando lectura previa")
        return cached

    # 2) PDF grande en páginas: partimos en paralelo desde el inicio.
    if total_pages > AZURE_FORCE_SPLIT_PAGES:
        _log(f"[Azure] PDF de {total_pages} págs. > umbral {AZURE_FORCE_SPLIT_PAGES}. Partiendo EN PARALELO.")
        result = _split_and_analyze_parallel(file_bytes, model, pages_per_chunk)
        _save_to_cache(original_hash, model, result)
        return result

    # 3) PDF chico: un solo intento directo.
    try:
        _log("[Azure] PDF chico. Intentando analizar PDF completo...")
        result = _analyze_single_pdf_bytes(file_bytes, model=model)
        _log("[Azure] PDF completo analizado correctamente.")
        write_progress(40, "Azure terminó la lectura", "documento leído")
        _save_to_cache(original_hash, model, result)
        return result

    except Exception as e:
        error_message = str(e)
        _log(f"[Azure] Error analizando PDF completo: {error_message}")
        if not _is_size_error(error_message):
            raise
        _log("[Azure] Azure rechazó el PDF por tamaño. Se dividirá EN PARALELO.")

    result = _split_and_analyze_parallel(file_bytes, model, pages_per_chunk)
    _save_to_cache(original_hash, model, result)
    return result