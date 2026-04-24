import json
import os
import re
from pathlib import Path
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

client = OpenAI(api_key=OPENAI_API_KEY)


def _clean_text(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    return text


def _extract_json_candidate(text: str) -> str:
    text = _clean_text(text)

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise Exception("OpenAI no devolvió un bloque JSON reconocible.")

    return text[start:end + 1]


def _parse_json_strict(text: str) -> Dict[str, Any]:
    candidate = _extract_json_candidate(text)
    return json.loads(candidate)


def _repair_json_with_openai(bad_text: str) -> Dict[str, Any]:
    prompt = f"""
Convierte el siguiente contenido en JSON válido.

Reglas:
- Responde SOLO JSON válido.
- No agregues explicación.
- No uses markdown.
- Conserva la estructura y el contenido lo más fiel posible.
- Si falta una coma, llave o corchete, repáralo.

Contenido:
{bad_text}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    repaired_text = response.output_text or ""
    repaired_candidate = _extract_json_candidate(repaired_text)
    return json.loads(repaired_candidate)


def _safe_json_from_model_output(output_text: str) -> Dict[str, Any]:
    try:
        return _parse_json_strict(output_text)
    except Exception:
        return _repair_json_with_openai(output_text)


def structure_licitacion_text(raw_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise Exception("Falta OPENAI_API_KEY en .env")

    prompt = f"""
Extrae del siguiente texto un JSON limpio y estructurado de una licitación pública.

Reglas:
- Responde SOLO JSON válido.
- No agregues comentarios.
- No agregues markdown.
- Si un dato no existe, usa null.
- Si no estás seguro, no inventes.
- "partidas" debe ir vacío si hay demasiadas.
- "fechas_clave" debe ser un arreglo.
- "anexos" debe ser un arreglo.
- "penalizaciones" debe ser un arreglo.
- "fuentes" puede ir vacío por ahora.

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
  "resumen": null,
  "fuentes": []
}}

Texto:
{raw_text}
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    output_text = response.output_text or ""
    return _safe_json_from_model_output(output_text)


def _normalize_number(value):
    if value is None:
        return None

    txt = str(value).strip()
    if not txt:
        return None

    txt = txt.replace(",", "")
    txt = txt.replace("O", "0")
    txt = txt.replace("o", "0")

    try:
        if "." in txt:
            return float(txt)
        return int(txt)
    except Exception:
        return txt


def _normalize_text(value):
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_yes_no(value):
    if value is None:
        return None

    txt = (_normalize_text(value) or "").lower()
    if txt in ["si", "sí", "s1", "sl"]:
        return "Si"
    if txt in ["no"]:
        return "No"
    return value


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


def _is_candidate_items_table(matrix: List[List[str]]) -> bool:
    flat = " ".join(" ".join(row) for row in matrix).lower()

    keywords = ["subpartida", "concepto", "unidad", "cantidad", "muestra"]
    score = sum(1 for k in keywords if k in flat)

    return score >= 3


def _extract_row_from_table(row: List[str]) -> Optional[Dict[str, Any]]:
    values = [_normalize_text(x) or "" for x in row]
    non_empty = [v for v in values if v]

    if len(non_empty) < 4:
        return None

    first = non_empty[0]
    if not re.match(r"^\d{1,4}$", first):
        return None

    subpartida = _normalize_number(first)

    unidad = None
    cantidad_minima = None
    cantidad_maxima = None
    muestra = None

    if len(non_empty) >= 6:
        unidad = non_empty[-4]
        cantidad_minima = _normalize_number(non_empty[-3])
        cantidad_maxima = _normalize_number(non_empty[-2])
        muestra = _normalize_yes_no(non_empty[-1])
        descripcion = " ".join(non_empty[1:-4]).strip()
    elif len(non_empty) == 5:
        unidad = non_empty[-3]
        cantidad_minima = _normalize_number(non_empty[-2])
        cantidad_maxima = _normalize_number(non_empty[-1])
        descripcion = " ".join(non_empty[1:-3]).strip()
    else:
        descripcion = " ".join(non_empty[1:]).strip()

    if not descripcion:
        return None

    return {
        "partida": 1,
        "subpartida": subpartida,
        "descripcion": descripcion,
        "unidad": unidad,
        "cantidad_minima": cantidad_minima,
        "cantidad_maxima": cantidad_maxima,
        "presentar_muestra": muestra,
    }


def extract_items_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    tables = raw_analyze_result.get("tables", []) or []

    all_items = []

    for table in tables:
        matrix = _build_table_matrix(table)

        if not _is_candidate_items_table(matrix):
            continue

        for row in matrix:
            parsed = _extract_row_from_table(row)
            if parsed:
                all_items.append(parsed)

    unique = []
    seen = set()

    for item in all_items:
        key = (
            str(item.get("subpartida") or "").strip(),
            str(item.get("descripcion") or "").strip().lower(),
        )

        if not key[0]:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    unique = sorted(unique, key=lambda x: int(x.get("subpartida") or 999999))

    return {
        "items_count": len(unique),
        "items": unique,
    }