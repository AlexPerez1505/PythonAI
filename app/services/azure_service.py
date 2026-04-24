import os
import sys
import time
from io import BytesIO
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _is_size_error(message: str) -> bool:
    text = (message or "").lower()
    return "invalidcontentlength" in text or "too large" in text


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
        raise Exception(f"Error iniciando Azure analysis: {response.status_code} - {response.text}")

    operation_location = response.headers.get("operation-location")
    if not operation_location:
        raise Exception("Azure no devolvió operation-location")

    while True:
        poll = requests.get(
            operation_location,
            headers={"Ocp-Apim-Subscription-Key": AZURE_KEY},
            timeout=180,
        )

        if not poll.ok:
            raise Exception(f"Error consultando Azure result: {poll.status_code} - {poll.text}")

        data = poll.json()
        status = data.get("status")

        if status in ["notStarted", "running"]:
            time.sleep(2)
            continue

        if status != "succeeded":
            raise Exception(f"Azure terminó con estado no exitoso: {data}")

        return data


def split_pdf_bytes(file_bytes: bytes, pages_per_chunk: int) -> List[bytes]:
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


def merge_azure_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged_content_parts = []
    merged_pages = []
    merged_tables = []
    current_page_offset = 0

    for result in results:
        analyze_result = result.get("analyzeResult", {})
        content = analyze_result.get("content", "")
        pages = analyze_result.get("pages", [])
        tables = analyze_result.get("tables", [])

        if content:
            merged_content_parts.append(content)

        for page in pages:
            page_copy = dict(page)
            if "pageNumber" in page_copy:
                page_copy["pageNumber"] = current_page_offset + page_copy["pageNumber"]
            merged_pages.append(page_copy)

        for table in tables:
            table_copy = dict(table)

            if "boundingRegions" in table_copy:
                new_regions = []
                for region in table_copy.get("boundingRegions", []) or []:
                    region_copy = dict(region)
                    if "pageNumber" in region_copy:
                        region_copy["pageNumber"] = current_page_offset + region_copy["pageNumber"]
                    new_regions.append(region_copy)
                table_copy["boundingRegions"] = new_regions

            new_cells = []
            for cell in table_copy.get("cells", []) or []:
                cell_copy = dict(cell)

                if "boundingRegions" in cell_copy:
                    cell_regions = []
                    for region in cell_copy.get("boundingRegions", []) or []:
                        region_copy = dict(region)
                        if "pageNumber" in region_copy:
                            region_copy["pageNumber"] = current_page_offset + region_copy["pageNumber"]
                        cell_regions.append(region_copy)
                    cell_copy["boundingRegions"] = cell_regions

                new_cells.append(cell_copy)

            table_copy["cells"] = new_cells
            merged_tables.append(table_copy)

        current_page_offset += len(pages)

    return {
        "status": "succeeded",
        "analyzeResult": {
            "content": "\n\n".join(merged_content_parts),
            "pages": merged_pages,
            "tables": merged_tables,
        },
    }


def analyze_pdf_with_auto_split(
    file_bytes: bytes,
    model: str = "prebuilt-layout",
    pages_per_chunk: int = 5,
) -> Dict[str, Any]:
    try:
        _log("Intentando analizar PDF completo...")
        return _analyze_single_pdf_bytes(file_bytes, model=model)
    except Exception as e:
        if not _is_size_error(str(e)):
            raise
        _log("Azure rechazó el PDF completo por tamaño. Se intentará dividir.")

    current_chunk_size = pages_per_chunk

    while current_chunk_size >= 1:
        _log(f"Intentando dividir PDF en bloques de {current_chunk_size} página(s)...")
        chunks = split_pdf_bytes(file_bytes, pages_per_chunk=current_chunk_size)
        partial_results = []
        failed_due_to_size = False

        for index, chunk_bytes in enumerate(chunks, start=1):
            try:
                _log(f"Procesando chunk {index}/{len(chunks)} con {current_chunk_size} página(s)...")
                partial_results.append(_analyze_single_pdf_bytes(chunk_bytes, model=model))
            except Exception as e:
                if _is_size_error(str(e)) and current_chunk_size > 1:
                    _log(f"Chunk {index} todavía muy grande. Se reducirá el tamaño del bloque.")
                    failed_due_to_size = True
                    break
                raise

        if not failed_due_to_size:
            return merge_azure_results(partial_results)

        current_chunk_size = current_chunk_size // 2

    raise Exception("No se pudo procesar el PDF: incluso dividido en bloques mínimos Azure lo rechazó por tamaño.")