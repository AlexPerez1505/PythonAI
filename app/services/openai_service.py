import json
import os
import re
from pathlib import Path
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)


def _openai_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Responde únicamente JSON válido. No uses markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content or ""


def _clean_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    text = text.replace("```json", "").replace("```", "").strip()
    return text


def _extract_json_candidate(text: str) -> str:
    text = _clean_text(text)

    start_obj = text.find("{")
    end_obj = text.rfind("}")

    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj:end_obj + 1]

    start_arr = text.find("[")
    end_arr = text.rfind("]")

    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr:end_arr + 1]

    raise Exception("OpenAI no devolvió un bloque JSON reconocible.")


def _parse_json_strict(text: str):
    candidate = _extract_json_candidate(text)
    return json.loads(candidate)


def _repair_json_with_openai(bad_text: str):
    prompt = f"""
Convierte el siguiente contenido en JSON válido.

Reglas:
- Responde SOLO JSON válido.
- No agregues explicación.
- No uses markdown.
- Conserva la estructura y el contenido.
- Si falta una coma, llave o corchete, repáralo.

Contenido:
{bad_text}
"""

    repaired_text = _openai_text(prompt)
    repaired_candidate = _extract_json_candidate(repaired_text)
    return json.loads(repaired_candidate)


def _safe_json_from_model_output(output_text: str):
    try:
        return _parse_json_strict(output_text)
    except Exception:
        return _repair_json_with_openai(output_text)


def structure_licitacion_text(raw_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise Exception("Falta OPENAI_API_KEY en .env")

    compact_text = (raw_text or "")[:45000]

    prompt = f"""
Extrae del siguiente texto un JSON limpio y estructurado de una licitación pública.

Reglas:
- Responde SOLO JSON válido.
- No agregues comentarios.
- No agregues markdown.
- Si un dato no existe, usa null.
- Si no estás seguro, no inventes.
- fechas_clave debe ser arreglo.
- anexos debe ser arreglo.
- penalizaciones debe ser arreglo.
- partidas puede ir vacío porque las partidas se extraen por otra función.

Estructura esperada:
{{
  "numero_procedimiento": null,
  "objeto": null,
  "dependencia": null,
  "tipo_procedimiento": null,
  "moneda": null,
  "anexos": [],
  "partidas": [],
  "fechas_clave": [],
  "penalizaciones": [],
  "condiciones_pago": null,
  "vigencia_contrato": null,
  "lugar_entrega": null,
  "plazo_entrega": null,
  "resumen": null,
  "fuentes": []
}}

Texto:
{compact_text}
"""

    output_text = _openai_text(prompt)
    return _safe_json_from_model_output(output_text)


def _normalize_text(value):
    if value is None:
        return None

    txt = re.sub(r"\s+", " ", str(value)).strip()
    return txt if txt else None


def _normalize_number(value):
    if value is None:
        return None

    txt = str(value).strip()
    if not txt:
        return None

    txt = txt.replace(",", "")
    txt = txt.replace(" ", "")
    txt = txt.replace("O", "0")
    txt = txt.replace("o", "0")

    try:
        if "." in txt:
            return float(txt)
        return int(txt)
    except Exception:
        return None


def _normalize_yes_no(value):
    if value is None:
        return None

    txt = (_normalize_text(value) or "").lower()

    if txt in ["si", "sí", "s1", "sl", "x", "aplica", "requiere"]:
        return "Si"

    if txt in ["no", "n/a", "na", "no aplica"]:
        return "No"

    return _normalize_text(value)


def _build_table_matrix(table: Dict[str, Any]) -> List[List[str]]:
    row_count = table.get("rowCount", 0)
    col_count = table.get("columnCount", 0)

    matrix = [["" for _ in range(col_count)] for _ in range(row_count)]

    for cell in table.get("cells", []) or []:
        r = cell.get("rowIndex", 0)
        c = cell.get("columnIndex", 0)
        content = _normalize_text(cell.get("content", "")) or ""

        if 0 <= r < row_count and 0 <= c < col_count:
            matrix[r][c] = content

    return matrix


def _header_key(text: str) -> str:
    txt = (text or "").lower()
    txt = txt.replace(".", "").replace(":", "").replace("#", "")
    txt = re.sub(r"\s+", " ", txt).strip()

    if any(k in txt for k in ["núm prog", "num prog", "no", "n°", "numero", "número", "partida", "subpartida", "renglon", "renglón"]):
        return "numero"

    if any(k in txt for k in ["cant min", "cantidad min", "mínima", "minima"]):
        return "cantidad_minima"

    if any(k in txt for k in ["cant max", "cantidad max", "máxima", "maxima"]):
        return "cantidad_maxima"

    if any(k in txt for k in ["cantidad", "cant", "volumen"]):
        return "cantidad"

    if any(k in txt for k in ["unidad de medida", "unidad", "u m", "u.m", "medida", "presentacion", "presentación"]):
        return "unidad"

    if any(k in txt for k in ["descripcion", "descripción", "concepto", "bien", "producto", "servicio", "nombre", "material", "articulo", "artículo"]):
        return "descripcion"

    if any(k in txt for k in ["muestra"]):
        return "muestra"

    return ""


def _looks_like_header(row: List[str]) -> bool:
    joined = " ".join(row).lower()

    words = [
        "partida", "subpartida", "núm", "num", "prog",
        "cantidad", "cant", "unidad", "medida",
        "descripcion", "descripción", "concepto", "bien",
        "producto", "servicio", "muestra",
    ]

    score = sum(1 for w in words if w in joined)
    return score >= 2


def _find_header_row(matrix: List[List[str]]) -> Optional[int]:
    best_index = None
    best_score = 0

    for i, row in enumerate(matrix[:8]):
        keys = [_header_key(x) for x in row]
        score = len([k for k in keys if k])

        if score > best_score:
            best_score = score
            best_index = i

    if best_score >= 2:
        return best_index

    return None


def _column_map_from_header(row: List[str]) -> Dict[str, int]:
    mapping = {}

    for idx, cell in enumerate(row):
        key = _header_key(cell)
        if key and key not in mapping:
            mapping[key] = idx

    return mapping


def _is_probable_item_row(row: List[str]) -> bool:
    values = [_normalize_text(x) or "" for x in row]
    non_empty = [v for v in values if v]

    if len(non_empty) < 2:
        return False

    first = non_empty[0]
    has_number_start = bool(re.match(r"^\d{1,5}$", first))
    has_long_text = any(len(v) >= 15 for v in non_empty)

    return has_number_start and has_long_text


def _extract_with_mapping(row: List[str], mapping: Dict[str, int]) -> Optional[Dict[str, Any]]:
    def get_col(key):
        idx = mapping.get(key)
        if idx is None or idx >= len(row):
            return None
        return _normalize_text(row[idx])

    numero = get_col("numero")
    numero_normalizado = _normalize_number(numero)

    descripcion = get_col("descripcion")
    unidad = get_col("unidad")
    cantidad_minima = _normalize_number(get_col("cantidad_minima"))
    cantidad_maxima = _normalize_number(get_col("cantidad_maxima"))
    cantidad = _normalize_number(get_col("cantidad"))
    muestra = _normalize_yes_no(get_col("muestra"))

    if not numero_normalizado:
        values = [_normalize_text(x) or "" for x in row]
        non_empty = [v for v in values if v]
        if non_empty and re.match(r"^\d{1,5}$", non_empty[0]):
            numero_normalizado = _normalize_number(non_empty[0])

    if not descripcion:
        used_indexes = set(mapping.values())
        leftovers = []

        for idx, value in enumerate(row):
            clean = _normalize_text(value)
            if not clean:
                continue
            if idx in used_indexes:
                continue
            if re.match(r"^\d{1,5}$", clean):
                continue
            leftovers.append(clean)

        descripcion = " ".join(leftovers).strip() if leftovers else None

    if not descripcion or len(descripcion) < 5:
        return None

    if not numero_normalizado:
        return None

    return {
        "partida": numero_normalizado,
        "subpartida": None,
        "numero": numero_normalizado,
        "descripcion": descripcion,
        "nombre": descripcion,
        "unidad": unidad,
        "cantidad": cantidad,
        "cantidad_minima": cantidad_minima,
        "cantidad_maxima": cantidad_maxima,
        "presentar_muestra": muestra,
    }


def _extract_heuristic(row: List[str]) -> Optional[Dict[str, Any]]:
    values = [_normalize_text(x) or "" for x in row]
    non_empty = [v for v in values if v]

    if len(non_empty) < 2:
        return None

    first = non_empty[0]

    if not re.match(r"^\d{1,5}$", first):
        return None

    numero = _normalize_number(first)

    numbers_after = []
    text_parts = []
    unidad = None
    muestra = None

    unidad_keywords = [
        "pieza", "pza", "pzas", "paquete", "caja", "bolsa", "rollo", "metro",
        "litro", "kg", "kilogramo", "servicio", "juego", "par", "lote",
        "unidad", "frasco", "bote", "sobre", "hoja", "block"
    ]

    for value in non_empty[1:]:
        value_clean = value.strip()
        num = _normalize_number(value_clean)

        if num is not None and re.match(r"^[\d,\.\sOo]+$", value_clean):
            numbers_after.append(num)
            continue

        low = value_clean.lower()

        if unidad is None and any(low == u or low.startswith(u + " ") for u in unidad_keywords):
            unidad = value_clean
            continue

        if muestra is None and low in ["si", "sí", "no", "s1", "sl", "x", "n/a", "na"]:
            muestra = _normalize_yes_no(value_clean)
            continue

        text_parts.append(value_clean)

    descripcion = " ".join(text_parts).strip()

    if not descripcion or len(descripcion) < 5:
        return None

    cantidad_minima = None
    cantidad_maxima = None
    cantidad = None

    if len(numbers_after) >= 2:
        cantidad_minima = numbers_after[0]
        cantidad_maxima = numbers_after[1]
    elif len(numbers_after) == 1:
        cantidad = numbers_after[0]

    return {
        "partida": numero,
        "subpartida": None,
        "numero": numero,
        "descripcion": descripcion,
        "nombre": descripcion,
        "unidad": unidad,
        "cantidad": cantidad,
        "cantidad_minima": cantidad_minima,
        "cantidad_maxima": cantidad_maxima,
        "presentar_muestra": muestra,
    }


def _extract_items_from_matrix(matrix: List[List[str]]) -> List[Dict[str, Any]]:
    items = []

    header_index = _find_header_row(matrix)

    if header_index is not None:
        mapping = _column_map_from_header(matrix[header_index])

        for row in matrix[header_index + 1:]:
            if _looks_like_header(row):
                continue

            parsed = _extract_with_mapping(row, mapping)
            if parsed:
                items.append(parsed)

    for row in matrix:
        if _looks_like_header(row):
            continue

        if not _is_probable_item_row(row):
            continue

        parsed = _extract_heuristic(row)
        if parsed:
            items.append(parsed)

    return items


def _extract_items_from_text(raw_text: str) -> List[Dict[str, Any]]:
    lines = []
    for line in (raw_text or "").splitlines():
        clean = _normalize_text(line)
        if clean:
            lines.append(clean)

    items = []
    i = 0

    unidad_words = r"(PIEZA|PIEZAS|PZA|PZAS|PAQUETE|CAJA|BOLSA|ROLLO|SERVICIO|JUEGO|PAR|LOTE|BOTE|FRASCO|HOJA|BLOCK|KG|LITRO|METRO)"

    while i < len(lines):
        line = lines[i]

        if re.match(r"^\d{1,5}$", line):
            numero = _normalize_number(line)
            chunk = lines[i:i + 16]

            nums = []
            unidad = None
            desc_parts = []

            for part in chunk[1:]:
                if re.match(r"^\d{1,5}$", part) and len(desc_parts) > 0:
                    break

                if _normalize_number(part) is not None and re.match(r"^[\d,\.\sOo]+$", part):
                    nums.append(_normalize_number(part))
                    continue

                if unidad is None and re.match(f"^{unidad_words}$", part.upper()):
                    unidad = part
                    continue

                if len(part) > 8:
                    desc_parts.append(part)

            descripcion = " ".join(desc_parts).strip()

            if numero and descripcion and len(descripcion) >= 10:
                item = {
                    "partida": numero,
                    "subpartida": None,
                    "numero": numero,
                    "descripcion": descripcion,
                    "nombre": descripcion,
                    "unidad": unidad,
                    "cantidad": nums[0] if len(nums) == 1 else None,
                    "cantidad_minima": nums[0] if len(nums) >= 2 else None,
                    "cantidad_maxima": nums[1] if len(nums) >= 2 else None,
                    "presentar_muestra": None,
                }
                items.append(item)

        i += 1

    return items


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for item in items:
        numero = str(item.get("numero") or item.get("partida") or item.get("subpartida") or "").strip()
        desc = str(item.get("descripcion") or item.get("nombre") or "").strip()

        if not numero or not desc:
            continue

        if numero not in merged:
            merged[numero] = item
            continue

        old_desc = str(merged[numero].get("descripcion") or merged[numero].get("nombre") or "")
        if len(desc) > len(old_desc):
            merged[numero] = item

    def sort_key(x):
        value = x.get("numero") or x.get("partida") or x.get("subpartida") or 999999
        try:
            return int(value)
        except Exception:
            return 999999

    return sorted(merged.values(), key=sort_key)


def extract_items_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    tables = raw_analyze_result.get("tables", []) or []
    all_items = []

    for table in tables:
        matrix = _build_table_matrix(table)
        all_items.extend(_extract_items_from_matrix(matrix))

    raw_text = raw_analyze_result.get("content", "") or ""
    all_items.extend(_extract_items_from_text(raw_text))

    unique = _dedupe_items(all_items)

    return {
        "items_count": len(unique),
        "items": unique,
    }