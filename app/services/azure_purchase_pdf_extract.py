import argparse
import json
import os
import re
import sys
import time
from io import BytesIO
from typing import Dict, Any, List, Optional

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

    response = requests.post(
        url,
        headers=headers,
        data=file_bytes,
        timeout=180,
    )

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
                _log(f"Procesando chunk {index}/{len(chunks)}...")
                partial_results.append(_analyze_single_pdf_bytes(chunk_bytes, model=model))

            except Exception as e:
                if _is_size_error(str(e)) and current_chunk_size > 1:
                    _log(f"Chunk {index} todavía muy grande. Reduciendo tamaño.")
                    failed_due_to_size = True
                    break

                raise

        if not failed_due_to_size:
            return merge_azure_results(partial_results)

        current_chunk_size = current_chunk_size // 2

    raise Exception("No se pudo procesar el PDF incluso dividiéndolo en páginas individuales.")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def money_to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    text = re.sub(r"[^0-9,.\-]", "", text)

    if not text:
        return None

    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")

    try:
        return round(float(text), 4)
    except Exception:
        return None


def looks_like_money_cell(value: str) -> bool:
    text = normalize_spaces(value)

    if not text:
        return False

    if re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", text):
        return False

    if "/" in text or '"' in text or "°" in text:
        return False

    if "$" in text:
        return True

    # 123.45, 1,234.56
    if re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2,4})", text):
        return True

    if re.fullmatch(r"[0-9]+(?:\.[0-9]{2,4})", text):
        return True

    return False


def extract_money_from_cell(value: str) -> Optional[float]:
    if not looks_like_money_cell(value):
        return None

    return money_to_float(value)


def find_money_values(text: str) -> List[float]:
    if not text:
        return []

    values = []

    parts = re.split(r"\s*\|\s*|\t+", text)

    for part in parts:
        amount = extract_money_from_cell(part)

        if amount is not None:
            values.append(amount)

    # Fallback solo para valores con símbolo $, no para medidas.
    if not values:
        matches = re.findall(
            r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2,4})|[0-9]+(?:\.[0-9]{2,4}))",
            text,
            flags=re.I,
        )

        for match in matches:
            value = money_to_float(match)

            if value is not None:
                values.append(value)

    return values


def parse_number(value: str) -> Optional[float]:
    text = normalize_spaces(value)

    if not text:
        return None

    if re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", text):
        return None

    if "/" in text or '"' in text or "°" in text:
        return None

    text = text.replace(",", "")

    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text):
        return None

    try:
        return float(text)
    except Exception:
        return None


def find_qty_values(text: str) -> Optional[float]:
    clean = normalize_spaces(text)

    qty_match = re.search(
        r"(?:cantidad|cant\.?|qty)\s*:?\s*([0-9]+(?:[.,][0-9]+)?)",
        clean,
        flags=re.I,
    )

    if qty_match:
        value = qty_match.group(1).replace(",", ".")

        try:
            qty = float(value)
            return qty if qty > 0 else None
        except Exception:
            return None

    first_number = re.match(r"^\s*([0-9]+(?:[.,][0-9]+)?)\s+", clean)

    if first_number:
        value = first_number.group(1).replace(",", ".")

        try:
            qty = float(value)

            if 0 < qty <= 999999:
                return qty

        except Exception:
            return None

    return None


def looks_like_unit(value: str) -> bool:
    text = normalize_spaces(value).upper().replace(".", "")

    units = {
        "PIEZA", "PZA", "PZ", "CAJA", "CAJAS", "PAQUETE", "PAQUETES",
        "PQT", "KG", "KILO", "KILOS", "GRAMO", "GRAMOS", "GR",
        "LITRO", "LITROS", "LT", "SERVICIO", "SERVICIOS", "M",
        "METRO", "METROS", "ROLLO", "ROLLOS", "BOLSA", "BOLSAS",
        "CUBETA", "CUBETAS", "JUEGO", "JUEGOS", "LOTE", "LOTES",
        "UNIDAD", "UNIDADES", "PAR", "PARES"
    }

    return text in units


def normalize_unit(value: str) -> str:
    text = normalize_spaces(value).upper().replace(".", "")

    aliases = {
        "PZA": "PIEZA",
        "PZ": "PIEZA",
        "UNIDAD": "PIEZA",
        "UNIDADES": "PIEZA",
        "CAJAS": "CAJA",
        "PAQUETES": "PAQUETE",
        "PQT": "PAQUETE",
        "KILO": "KG",
        "KILOS": "KG",
        "GRAMOS": "GR",
        "LITROS": "LT",
        "SERVICIOS": "SERVICIO",
        "METRO": "M",
        "METROS": "M",
        "ROLLOS": "ROLLO",
        "BOLSAS": "BOLSA",
        "CUBETAS": "CUBETA",
        "JUEGOS": "JUEGO",
        "LOTES": "LOTE",
        "PARES": "PAR",
    }

    return aliases.get(text, text)


def looks_like_description(value: str) -> bool:
    text = normalize_spaces(value)

    if len(text) < 10:
        return False

    if not re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", text):
        return False

    if looks_like_unit(text):
        return False

    bad = text.lower()

    if any(word in bad for word in [
        "subtotal",
        "total",
        "iva",
        "impuesto",
        "sello",
        "uuid",
        "certificado",
        "forma de pago",
        "método de pago",
        "metodo de pago",
        "emisor",
        "receptor",
        "rfc",
        "régimen fiscal",
        "regimen fiscal",
        "lugar de expedición",
        "lugar de expedicion",
    ]):
        return False

    return True


def choose_description_from_values(values: List[str]) -> str:
    descriptions = [v for v in values if looks_like_description(v)]

    if not descriptions:
        return ""

    return max(descriptions, key=len)[:255]


def choose_unit_from_values(values: List[str]) -> str:
    for value in values:
        if looks_like_unit(value):
            return normalize_unit(value)

    return ""


def detect_document_header(content: str, category: str) -> Dict[str, Any]:
    text = normalize_spaces(content)
    lower = text.lower()

    uuid = None
    uuid_match = re.search(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        text,
    )

    if uuid_match:
        uuid = uuid_match.group(0).upper()

    rfc = None
    rfc_match = re.search(
        r"\b[A-ZÑ&]{3,4}[0-9]{6}[A-Z0-9]{3}\b",
        text,
        flags=re.I,
    )

    if rfc_match:
        rfc = rfc_match.group(0).upper()

    currency = "MXN"

    if re.search(r"\bUSD\b|d[oó]lares|dollars", text, flags=re.I):
        currency = "USD"

    document_type = "otro"

    if "factura" in lower or "cfdi" in lower or uuid:
        document_type = "factura"
    elif "remision" in lower or "remisión" in lower:
        document_type = "remision"
    elif "ticket" in lower or "recibo" in lower:
        document_type = "ticket"

    date_value = None

    date_match = re.search(
        r"(\d{4}[-/]\d{2}[-/]\d{2})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?",
        text,
    )

    if not date_match:
        date_match = re.search(
            r"(\d{2}[-/]\d{2}[-/]\d{4})(?:[ T](\d{2}:\d{2}(?::\d{2})?))?",
            text,
        )

    if date_match:
        raw_date = date_match.group(1).replace("/", "-")
        raw_time = date_match.group(2) or "00:00:00"

        if len(raw_time) == 5:
            raw_time += ":00"

        parts = raw_date.split("-")

        if len(parts[0]) == 4:
            date_value = f"{raw_date} {raw_time}"
        else:
            date_value = f"{parts[2]}-{parts[1]}-{parts[0]} {raw_time}"

    subtotal = 0
    tax = 0
    total = 0

    subtotal_match = re.search(
        r"subtotal\s*:?\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        flags=re.I,
    )

    tax_match = re.search(
        r"\b(?:iva|impuesto|tax)\b\s*:?\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        flags=re.I,
    )

    total_matches = re.findall(
        r"\btotal\b\s*:?\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)",
        text,
        flags=re.I,
    )

    if subtotal_match:
        subtotal = money_to_float(subtotal_match.group(1)) or 0

    if tax_match:
        tax = money_to_float(tax_match.group(1)) or 0

    if total_matches:
        total = money_to_float(total_matches[-1]) or 0

    supplier_name = None
    lines = [normalize_spaces(line) for line in content.splitlines() if normalize_spaces(line)]

    for line in lines[:60]:
        low = line.lower()

        if any(skip in low for skip in [
            "factura",
            "cfdi",
            "rfc",
            "fecha",
            "folio",
            "uuid",
            "total",
            "subtotal",
            "certificado",
            "sello",
            "cadena original",
            "gobierno de méxico",
            "gobierno de mexico",
        ]):
            continue

        if len(line) >= 4 and re.search(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]", line):
            supplier_name = line[:180]
            break

    return {
        "document_type": document_type,
        "supplier_name": supplier_name,
        "counterparty_rfc": rfc,
        "uuid": uuid,
        "serie": None,
        "folio": None,
        "currency": currency,
        "document_datetime": date_value,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
    }


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


def row_is_noise(line: str) -> bool:
    low = line.lower()

    if not line or len(line) < 4:
        return True

    return any(word in low for word in [
        "subtotal",
        "total",
        "iva",
        "impuesto",
        "tax",
        "forma de pago",
        "metodo de pago",
        "método de pago",
        "sello",
        "cadena original",
        "uuid",
        "certificado",
        "emisor",
        "receptor",
        "regimen fiscal",
        "régimen fiscal",
        "lugar de expedición",
        "lugar de expedicion",
    ])


def clean_item_name_from_line(line: str) -> str:
    item_name = re.sub(
        r"\$\s*[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2,4})|\$\s*[0-9]+(?:\.[0-9]{2,4})",
        " ",
        line,
    )

    item_name = re.sub(
        r"\b(?:cantidad|cant\.?|qty|precio|importe|unidad|clave|producto|servicio|descripcion|descripción)\b",
        " ",
        item_name,
        flags=re.I,
    )

    item_name = re.sub(r"[\|\:\-]+", " ", item_name)
    item_name = normalize_spaces(item_name)

    return item_name[:255]


def infer_qty_from_values(values: List[str], description: str, unit: str) -> float:
    numeric_cells = []

    for value in values:
        text = normalize_spaces(value)

        if text == description:
            continue

        if unit and normalize_unit(text) == unit:
            continue

        n = parse_number(text)

        if n is None:
            continue

        # Evita agarrar decimales/medidas como cantidad.
        if n <= 0:
            continue

        numeric_cells.append(n)

    if not numeric_cells:
        return 1.0

    # En muchos formatos:
    # partida | cantidad | clave | unidad | descripción
    # Si hay varios enteros, el segundo suele ser cantidad.
    if len(numeric_cells) >= 2:
        qty = numeric_cells[1]
    else:
        qty = numeric_cells[0]

    if qty <= 0:
        return 1.0

    return round(qty, 3)


def infer_amounts_from_values(values: List[str], qty: float) -> Dict[str, Optional[float]]:
    amount_cells = []

    for value in values:
        amount = extract_money_from_cell(value)

        if amount is not None and amount > 0:
            amount_cells.append(amount)

    unit_price = None
    line_total = None

    if len(amount_cells) >= 2:
        unit_price = amount_cells[-2]
        line_total = amount_cells[-1]
    elif len(amount_cells) == 1:
        line_total = amount_cells[0]

        if qty > 0:
            unit_price = round(line_total / qty, 4)

    return {
        "unit_price": unit_price,
        "line_total": line_total,
    }


def extract_items_from_tables(tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []

    for table in tables:
        matrix = table_to_matrix(table)

        for row in matrix:
            values = [normalize_spaces(v) for v in row if normalize_spaces(v)]

            if not values:
                continue

            line = " | ".join(values)

            if row_is_noise(line):
                continue

            item_name = choose_description_from_values(values)

            if not item_name:
                continue

            unit = choose_unit_from_values(values)
            qty = infer_qty_from_values(values, item_name, unit)

            amounts = infer_amounts_from_values(values, qty)

            unit_price = amounts["unit_price"]
            line_total = amounts["line_total"]

            items.append({
                "item_raw": line,
                "item_name": item_name[:255],
                "qty": qty,
                "unit": unit,
                "unit_price": unit_price,
                "line_total": line_total,
                "ai_meta": {
                    "prodserv": None,
                    "source": "azure_document_intelligence_table",
                    "has_amounts": bool(line_total and line_total > 0),
                },
            })

    return items


def extract_items_from_lines(content: str) -> List[Dict[str, Any]]:
    items = []

    for raw_line in content.splitlines():
        line = normalize_spaces(raw_line)

        if row_is_noise(line):
            continue

        if not looks_like_description(line):
            continue

        money_values = find_money_values(line)
        qty = find_qty_values(line) or 1.0

        unit_price = None
        line_total = None

        if len(money_values) >= 2:
            unit_price = money_values[-2]
            line_total = money_values[-1]
        elif len(money_values) == 1:
            line_total = money_values[-1]
            unit_price = round(line_total / qty, 4) if qty > 0 else None

        item_name = clean_item_name_from_line(line)

        if not item_name:
            continue

        items.append({
            "item_raw": line,
            "item_name": item_name[:255],
            "qty": qty,
            "unit": "",
            "unit_price": unit_price,
            "line_total": line_total,
            "ai_meta": {
                "prodserv": None,
                "source": "azure_document_intelligence_content",
                "has_amounts": bool(line_total and line_total > 0),
            },
        })

    return items


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for item in items:
        name = normalize_spaces(item.get("item_name", "")).lower()

        if not name:
            continue

        key = (
            name,
            str(item.get("qty", "")),
            str(item.get("unit", "")),
            str(item.get("unit_price", "")),
            str(item.get("line_total", "")),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def convert_azure_result_to_purchase_json(azure_result: Dict[str, Any], category: str) -> Dict[str, Any]:
    analyze_result = azure_result.get("analyzeResult", {})

    content = analyze_result.get("content", "") or ""
    tables = analyze_result.get("tables", []) or []
    pages = analyze_result.get("pages", []) or []

    document = detect_document_header(content, category)

    items = extract_items_from_tables(tables)

    if not items:
        items = extract_items_from_lines(content)

    items = dedupe_items(items)

    items_with_amounts = [
        i for i in items
        if i.get("line_total") is not None and float(i.get("line_total") or 0) > 0
    ]

    if document.get("total", 0) <= 0 and items_with_amounts:
        document["total"] = round(
            sum(float(i.get("line_total") or 0) for i in items_with_amounts),
            2,
        )

    warnings = []

    if not items:
        warnings.append("Azure Document Intelligence no detectó conceptos.")

    if items and not items_with_amounts:
        warnings.append("Se detectaron conceptos, pero no se detectaron importes confiables.")

    if not content:
        warnings.append("Azure Document Intelligence no devolvió contenido textual.")

    return {
        "document": document,
        "items": items,
        "notes": {
            "warnings": warnings,
            "confidence": 0.85 if items else 0.35,
            "engine": "azure_document_intelligence",
            "pages": len(pages),
            "tables": len(tables),
            "items_count": len(items),
            "items_with_amounts": len(items_with_amounts),
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file", required=True)
    parser.add_argument("--category", default="compra")
    parser.add_argument("--model", default="prebuilt-layout")
    parser.add_argument("--pages-per-chunk", type=int, default=5)
    parser.add_argument("--raw", action="store_true")

    args = parser.parse_args()

    if not os.path.isfile(args.file):
        raise Exception(f"No existe el archivo PDF: {args.file}")

    with open(args.file, "rb") as f:
        file_bytes = f.read()

    azure_result = analyze_pdf_with_auto_split(
        file_bytes=file_bytes,
        model=args.model,
        pages_per_chunk=args.pages_per_chunk,
    )

    if args.raw:
        print(json.dumps(azure_result, ensure_ascii=False))
        return

    normalized = convert_azure_result_to_purchase_json(
        azure_result=azure_result,
        category=args.category,
    )

    print(json.dumps(normalized, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print(
            json.dumps({"error": str(e)}, ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(1)