import argparse
import json
import os
import re
import sys
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from pypdf import PdfReader, PdfWriter

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").rstrip("/")
AZURE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_VERSION", "2024-11-30")

# Paralelización de bloques (tuneable por .env)
AZURE_MAX_WORKERS = int(os.getenv("AZURE_MAX_WORKERS", "8"))
AZURE_PAGES_PER_CHUNK = int(os.getenv("AZURE_PAGES_PER_CHUNK", "10"))


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Azure Document Intelligence
# ---------------------------------------------------------------------------

def _is_size_error(message: str) -> bool:
    text = (message or "").lower()
    return (
        "invalidcontentlength" in text
        or "too large" in text
        or "request body is too large" in text
        or "content length" in text
        or "maximum request body size" in text
    )


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
            time.sleep(1)
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
                page_copy["pageNumber"] = current_page_offset + int(page_copy["pageNumber"])
            merged_pages.append(page_copy)

        for table in tables:
            table_copy = dict(table)

            if "boundingRegions" in table_copy:
                new_regions = []
                for region in table_copy.get("boundingRegions", []) or []:
                    region_copy = dict(region)
                    if "pageNumber" in region_copy:
                        region_copy["pageNumber"] = current_page_offset + int(region_copy["pageNumber"])
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
                            region_copy["pageNumber"] = current_page_offset + int(region_copy["pageNumber"])
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


def _analyze_chunks_parallel(chunks: List[bytes], model: str, max_workers: int):
    """Procesa los bloques EN PARALELO pero devuelve los resultados EN ORDEN."""
    results: List[Optional[Dict[str, Any]]] = [None] * len(chunks)
    errors: Dict[int, Exception] = {}
    workers = max(1, min(max_workers, len(chunks)))

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
            except Exception as e:
                errors[idx] = e
            done += 1
            _log(f"Bloque Azure {done}/{len(chunks)} listo")

    return results, errors


def analyze_pdf_with_auto_split(
    file_bytes: bytes,
    model: str = "prebuilt-layout",
    pages_per_chunk: int = None,
) -> Dict[str, Any]:
    if pages_per_chunk is None:
        pages_per_chunk = AZURE_PAGES_PER_CHUNK

    try:
        _log("Analizando PDF con Azure Document Intelligence...")
        return _analyze_single_pdf_bytes(file_bytes, model=model)

    except Exception as e:
        if not _is_size_error(str(e)):
            raise
        _log("Azure rechazó el PDF completo por tamaño. Se dividirá EN PARALELO.")

    current_chunk_size = pages_per_chunk

    while current_chunk_size >= 1:
        _log(f"Dividiendo en bloques de {current_chunk_size} pág. (paralelo, {AZURE_MAX_WORKERS} workers)...")

        chunks = split_pdf_bytes(file_bytes, pages_per_chunk=current_chunk_size)

        results, errors = _analyze_chunks_parallel(chunks, model, AZURE_MAX_WORKERS)

        # ¿Algún error que NO sea de tamaño? -> se propaga tal cual.
        other_error = next((e for e in errors.values() if not _is_size_error(str(e))), None)
        if other_error is not None:
            raise other_error

        # ¿Algún bloque falló por tamaño? -> partir más chico y reintentar.
        size_error = any(_is_size_error(str(e)) for e in errors.values())

        if size_error:
            if current_chunk_size > 1:
                current_chunk_size = current_chunk_size // 2
                continue
            raise Exception("No se pudo procesar el PDF incluso dividiéndolo en páginas individuales.")

        return merge_azure_results(results)

    raise Exception("No se pudo procesar el PDF incluso dividiéndolo en páginas individuales.")


# ---------------------------------------------------------------------------
# Helpers mínimos compartidos
# ---------------------------------------------------------------------------

def normalize_spaces(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def table_to_matrix(table: Dict[str, Any]) -> List[List[str]]:
    rows = int(table.get("rowCount", 0) or 0)
    cols = int(table.get("columnCount", 0) or 0)

    matrix = [["" for _ in range(cols)] for _ in range(rows)]

    for cell in table.get("cells", []) or []:
        r = int(cell.get("rowIndex", 0) or 0)
        c = int(cell.get("columnIndex", 0) or 0)
        rs = int(cell.get("rowSpan", 1) or 1)
        cs = int(cell.get("columnSpan", 1) or 1)
        content = normalize_spaces(cell.get("content", ""))

        for rr in range(r, min(r + rs, rows)):
            for cc in range(c, min(c + cs, cols)):
                if not matrix[rr][cc]:
                    matrix[rr][cc] = content

    return matrix


def _flatten_azure_tables(tables: List[Dict[str, Any]]) -> str:
    out = []

    for ti, table in enumerate(tables, start=1):
        matrix = table_to_matrix(table)

        if not matrix:
            continue

        rows = []
        for row in matrix:
            cells = [normalize_spaces(c) for c in row]
            rows.append(" | ".join(cells))

        out.append(f"--- TABLA {ti} ({len(matrix)} filas) ---\n" + "\n".join(rows))

    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def _resolve_openai_base_url() -> str:
    """
    Resuelve la URL base que va a usar el cliente de OpenAI.

    Si OPENAI_BASE_URL viene seteada (a veces se hereda de PHP-FPM con un valor
    que NO termina en /v1, como 'https://api.openai.com'), la corregimos para que
    SIEMPRE termine en /v1. Si no viene, usamos el default oficial.
    """
    base = os.getenv("OPENAI_BASE_URL", "").strip()

    if not base:
        return DEFAULT_OPENAI_BASE_URL

    base = base.rstrip("/")

    if not base.endswith("/v1"):
        base = base + "/v1"

    return base


# ---------------------------------------------------------------------------
# OpenAI: la IA hace TODO el trabajo de identificar productos vs basura
# ---------------------------------------------------------------------------

OPENAI_SYSTEM_PROMPT = """Eres un extractor experto de documentos contables mexicanos (CFDI 4.0, facturas, tickets, remisiones, notas de venta, recibos de honorarios, comprobantes de cualquier proveedor).

Tu tarea es leer lo que Azure Document Intelligence extrajo de un PDF y devolver un JSON ESTRICTO con:
1. Los datos generales del documento.
2. ÚNICAMENTE las líneas reales de productos o servicios facturados.

Tú ya sabes cómo se ve una factura mexicana sin que te lo digan. Aplica ese criterio.

REGLAS DE NEGOCIO:
- Devuelve SOLO los renglones que son productos/servicios reales facturados al cliente.
- IGNORA TODO lo que sea: subtotales, totales, IVA, ISR, IEPS, retenciones, traslados, monto del impuesto, importe en letra ("NUEVE MIL QUINIENTOS PESOS"), sellos digitales, cadenas originales, datos del emisor/receptor, RFCs sueltos, fechas, números de serie, claves SAT sueltas, leyendas legales, condiciones de pago, observaciones, encabezados de tabla.
- Si una celda mezcla descripción del producto con basura del pie (códigos SAT pegados, "Traslado Base", texto del importe en letras, etc.), QUEDATE SOLO con la descripción real del producto.
- qty × unit_price debe coincidir con line_total (tolerancia ±2%). Si no coincide, ajusta unit_price tomando line_total como verdadero.
- Si la factura tiene 1 línea real de producto, devuelve UN solo item. Si tiene 50, devuelve los 50. NO inventes filas extra ni dupliques.
- Conserva acentos y mayúsculas/minúsculas naturales del nombre del producto.
- document_datetime en formato "YYYY-MM-DD HH:MM:SS". Si solo hay fecha sin hora, usa "YYYY-MM-DD 00:00:00".
- Para campos de item que no veas con claridad, usa null (NO 0).
- Para subtotal/tax/total del document, usa 0 si no aparecen.
- unit debe ser la unidad real del producto (PIEZA, CAJA, KG, LT, SERVICIO, etc.), no códigos SAT como "H87".

FORMATO JSON ESTRICTO (responde SOLO esto, sin markdown, sin ```):
{
  "document": {
    "document_type": "factura|ticket|remision|nota|recibo|otro",
    "supplier_name": null,
    "counterparty_rfc": null,
    "uuid": null,
    "serie": null,
    "folio": null,
    "currency": "MXN",
    "document_datetime": null,
    "subtotal": 0,
    "tax": 0,
    "total": 0
  },
  "items": [
    {
      "item_name": "descripción real del producto o servicio",
      "qty": 1,
      "unit": "PIEZA",
      "unit_price": 0,
      "line_total": 0,
      "prodserv_code": null
    }
  ]
}"""


def structure_with_openai(azure_result: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Manda lo que Azure extrajo a OpenAI y deja que la IA decida qué es producto."""
    try:
        from openai import OpenAI
        import openai as openai_pkg
    except ImportError:
        raise Exception("El paquete 'openai' no está instalado. Ejecuta: pip3 install --user openai")

    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    base_url = _resolve_openai_base_url()

    # ── DEBUG ──
    _log(f"[DEBUG] openai package version: {getattr(openai_pkg, '__version__', 'unknown')}")
    _log(f"[DEBUG] OPENAI_API_KEY recibida (primeros 10 chars): {api_key[:10] if api_key else '(VACIA)'}")
    _log(f"[DEBUG] OPENAI_MODEL recibido: {model!r}")
    _log(f"[DEBUG] OPENAI_BASE_URL final usado: {base_url!r}")
    _log(f"[DEBUG] OPENAI_BASE_URL crudo del entorno: {os.getenv('OPENAI_BASE_URL', '(no set)')!r}")
    _log(f"[DEBUG] Python: {sys.version.split()[0]}")
    # ──────────

    if not api_key:
        raise Exception("Falta OPENAI_API_KEY en el entorno.")

    analyze_result = azure_result.get("analyzeResult", {})
    content = (analyze_result.get("content", "") or "")[:40000]
    tables = analyze_result.get("tables", []) or []
    tables_text = _flatten_azure_tables(tables)[:40000]

    party_hint = (
        "El documento es una VENTA emitida por nosotros. El cliente / receptor es la contraparte (NO nosotros)."
        if category == "venta"
        else "El documento es una COMPRA que nos hicieron. El proveedor / emisor es la contraparte."
    )

    user_prompt = f"""{party_hint}

=== TEXTO CRUDO DEL DOCUMENTO ===
{content}

=== TABLAS DETECTADAS POR AZURE ===
{tables_text or "(Azure no detectó tablas estructuradas)"}
"""

    # Forzamos base_url para que NUNCA herede un valor mal formado de PHP-FPM
    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        body = getattr(e, "body", None)
        _log(f"[DEBUG] OpenAI exception type: {type(e).__name__}")
        _log(f"[DEBUG] OpenAI exception str: {e}")
        if body is not None:
            _log(f"[DEBUG] OpenAI exception body: {body}")
        raise

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def convert_openai_result_to_purchase_json(
    openai_result: Dict[str, Any],
    azure_result: Dict[str, Any],
    category: str,
) -> Dict[str, Any]:
    """Adapta el JSON de OpenAI al formato que espera PHP."""
    doc_in = openai_result.get("document", {}) or {}
    items_in = openai_result.get("items", []) or []

    items_out = []

    for it in items_in:
        if not isinstance(it, dict):
            continue

        name = normalize_spaces(it.get("item_name") or "")
        if not name:
            continue

        qty_raw = it.get("qty")
        try:
            qty = float(qty_raw) if qty_raw is not None else 1.0
            if qty <= 0:
                qty = 1.0
        except (TypeError, ValueError):
            qty = 1.0

        unit_price_raw = it.get("unit_price")
        line_total_raw = it.get("line_total")

        try:
            unit_price = float(unit_price_raw) if unit_price_raw not in (None, "") else None
        except (TypeError, ValueError):
            unit_price = None

        try:
            line_total = float(line_total_raw) if line_total_raw not in (None, "") else None
        except (TypeError, ValueError):
            line_total = None

        if line_total is None and unit_price is not None and qty > 0:
            line_total = round(unit_price * qty, 2)

        if unit_price is None and line_total is not None and qty > 0:
            unit_price = round(line_total / qty, 4)

        unit = normalize_spaces(it.get("unit") or "PIEZA").upper() or "PIEZA"

        items_out.append({
            "item_raw": name,
            "item_name": name[:255],
            "qty": round(qty, 3),
            "unit": unit,
            "unit_price": unit_price,
            "line_total": line_total,
            "ai_meta": {
                "prodserv": it.get("prodserv_code"),
                "source": "azure_document_intelligence_openai",
                "has_amounts": bool(line_total and line_total > 0),
            },
        })

    rfc_raw = doc_in.get("counterparty_rfc")
    uuid_raw = doc_in.get("uuid")

    document = {
        "document_type": doc_in.get("document_type") or "factura",
        "supplier_name": doc_in.get("supplier_name"),
        "counterparty_rfc": rfc_raw.upper() if isinstance(rfc_raw, str) and rfc_raw.strip() else None,
        "uuid": uuid_raw.upper() if isinstance(uuid_raw, str) and uuid_raw.strip() else None,
        "serie": doc_in.get("serie"),
        "folio": doc_in.get("folio"),
        "currency": doc_in.get("currency") or "MXN",
        "document_datetime": doc_in.get("document_datetime"),
        "subtotal": float(doc_in.get("subtotal") or 0),
        "tax": float(doc_in.get("tax") or 0),
        "total": float(doc_in.get("total") or 0),
    }

    items_with_amounts = [
        i for i in items_out
        if i.get("line_total") and float(i["line_total"]) > 0
    ]

    if document["subtotal"] <= 0 and items_with_amounts:
        document["subtotal"] = round(
            sum(float(i["line_total"]) for i in items_with_amounts), 2
        )

    if document["tax"] <= 0 and document["subtotal"] > 0:
        document["tax"] = round(document["subtotal"] * 0.16, 2)

    if document["total"] <= 0 and document["subtotal"] > 0:
        document["total"] = round(document["subtotal"] + document["tax"], 2)

    analyze_result = azure_result.get("analyzeResult", {})
    pages = analyze_result.get("pages", []) or []
    tables = analyze_result.get("tables", []) or []

    warnings = []
    if not items_out:
        warnings.append("La IA no identificó productos en el documento.")

    return {
        "document": document,
        "items": items_out,
        "notes": {
            "warnings": warnings,
            "confidence": 0.92 if items_out else 0.40,
            "engine": "azure_document_intelligence + openai",
            "openai_model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            "pages": len(pages),
            "tables": len(tables),
            "items_count": len(items_out),
            "items_with_amounts": len(items_with_amounts),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--category", default="compra")
    parser.add_argument("--model", default="prebuilt-layout")
    parser.add_argument("--pages-per-chunk", type=int, default=None)
    parser.add_argument("--raw", action="store_true", help="Devuelve respuesta cruda de Azure (debug)")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise Exception(f"No existe el archivo PDF: {args.file}")

    with open(args.file, "rb") as file:
        file_bytes = file.read()

    azure_result = analyze_pdf_with_auto_split(
        file_bytes=file_bytes,
        model=args.model,
        pages_per_chunk=args.pages_per_chunk,
    )

    if args.raw:
        print(json.dumps(azure_result, ensure_ascii=False))
        return

    openai_result = structure_with_openai(azure_result, args.category)

    final = convert_openai_result_to_purchase_json(
        openai_result=openai_result,
        azure_result=azure_result,
        category=args.category,
    )

    print(json.dumps(final, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(
            json.dumps({"error": str(e)}, ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(1)