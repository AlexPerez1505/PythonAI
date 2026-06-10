import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

client = OpenAI(api_key=OPENAI_API_KEY)


def _openai_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Responde únicamente JSON válido. No uses markdown."},
            {"role": "user", "content": prompt},
        ],
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


def _safe_json(text: str):
    return json.loads(_extract_json_candidate(text))


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
    try:
        return _safe_json(_openai_text(prompt))
    except Exception:
        return {"partidas": []}


def _normalize_text(value):
    if value is None:
        return None
    txt = re.sub(r"\s+", " ", str(value)).strip()
    return txt if txt else None


def _normalize_number(value):
    if value is None:
        return None
    txt = str(value).strip()
    if txt == "":
        return None
    txt = txt.replace(",", "").replace(" ", "")
    if not re.match(r"^-?\d+(\.\d+)?$", txt):
        return None
    try:
        return float(txt) if "." in txt else int(txt)
    except Exception:
        return None


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


def _matrix_to_rows(matrix: List[List[str]]) -> List[str]:
    rows = []
    for row in matrix:
        cells = [(c or "").strip() for c in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


# ===================== EXTRACCIÓN CON OPENAI (cualquier formato) =====================

_PROMPT_SISTEMA_ITEMS = (
    "Eres un extractor experto de partidas de anexos de licitaciones públicas de gobierno en México. "
    "Te paso filas de una tabla (cada fila trae sus celdas separadas por ' | '). "
    "Tu trabajo es devolver ÚNICAMENTE los renglones que sean un BIEN o SERVICIO concreto que una "
    "empresa pueda cotizar/comprar (ej. 'Arillo metálico doble 3/8\"', 'Block con 100 notas autoadheribles', "
    "'Servicio de mantenimiento de aire acondicionado').\n\n"
    "REGLAS DE VALIDACIÓN (muy importante):\n"
    "- Un renglón VÁLIDO describe un producto/servicio real. Si una fila NO lo es, OMÍTELA por completo.\n"
    "- NUNCA devuelvas como producto: encabezados de columna ('Descripción', 'Cantidad', 'Unidad', "
    "'Costo unitario antes de IVA', 'Precio', 'Partida'), números de página/hoja ('1 de 16', 'Hoja 3', 'HOJA'), "
    "etiquetas de formulario (FECHA, HORARIO, DOMICILIO, LUGAR, NOMBRE), fechas u horarios "
    "('8:30 a 13:00 horas'), domicilios o direcciones, nombres de dependencias o áreas, "
    "totales/subtotales, firmas, ni notas o instrucciones.\n"
    "- Si TODA la tabla es una portada, formulario, calendario, domicilios o cláusulas (no una lista de "
    "bienes/servicios), devuelve {\"items\":[]}.\n\n"
    "CAMPOS por cada producto válido:\n"
    "- 'descripcion': el NOMBRE/DESCRIPCIÓN real del producto. NUNCA pongas aquí el código o la clave.\n"
    "- 'clave': el código/clave de partida presupuestal si existe (ej. '21101-0106'); si no, null.\n"
    "- 'partida': número de partida si existe (entero). Si la numeración es jerárquica '1.1', usa partida=1 y subpartida='1'. Si no hay, null.\n"
    "- 'subpartida': solo para numeración jerárquica (lo que va después del punto); si no aplica, null.\n"
    "- 'unidad': unidad de medida (PIEZA, CAJA, PAQUETE, KG...). Si cantidad y unidad vienen juntas "
    "('10 caja/paquete 90 piezas'), pon cantidad=10 y unidad='CAJA'.\n"
    "- 'cantidad': cantidad solicitada (número). 'cantidad_minima'/'cantidad_maxima' si trae mínimo y máximo. "
    "OJO: la columna de 'Partida'/'No.' (1, 2, 3...) NO es la cantidad; si no hay cantidad real, deja cantidad en null.\n"
    "- 'muestra': 'Si'/'No' si la tabla lo indica; si no, null.\n\n"
    "Devuelve SOLO JSON con esta forma EXACTA: "
    "{\"items\":[{\"partida\":null,\"subpartida\":null,\"clave\":null,\"descripcion\":\"\",\"unidad\":\"\",\"cantidad\":null,\"cantidad_minima\":null,\"cantidad_maxima\":null,\"muestra\":null}]}"
)


def _openai_extract_chunk(rows: List[str]) -> List[Dict[str, Any]]:
    bloque = "\n".join(rows)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _PROMPT_SISTEMA_ITEMS},
                {"role": "user", "content": "Filas de la tabla:\n\n" + bloque},
            ],
        )
        data = json.loads(_extract_json_candidate(resp.choices[0].message.content or ""))
        items = data.get("items") or []
        return items if isinstance(items, list) else []
    except Exception as e:
        # IMPORTANTE: por stderr, NUNCA por stdout (rompería el JSON que lee Laravel).
        print(f"[extract_items] ERROR OpenAI con modelo '{OPENAI_MODEL}': {e}", file=sys.stderr, flush=True)
        return []


def extract_items_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    tables = raw_analyze_result.get("tables", []) or []
    api_key = os.getenv("OPENAI_API_KEY", "")

    crudos_all: List[Dict[str, Any]] = []

    if api_key and tables:
        for table in tables:
            rows = _matrix_to_rows(_build_table_matrix(table))
            if len(rows) < 2:
                continue
            # Extrae por lotes de 45 filas (tablas que abarcan varias hojas).
            for i in range(0, len(rows), 45):
                crudos_all.extend(_openai_extract_chunk(rows[i:i + 45]))

    if crudos_all:
        out = []
        seen = set()
        seq = 0
        for it in crudos_all:
            if not isinstance(it, dict):
                continue
            desc = _normalize_text(it.get("descripcion") or it.get("nombre"))
            if not desc or len(desc) < 3:
                continue

            clave = _normalize_text(it.get("clave"))
            sub = _normalize_text(it.get("subpartida"))
            cant = _normalize_number(it.get("cantidad"))

            key = ((clave or ""), desc.lower()[:160], str(cant or ""), (sub or ""))
            if key in seen:
                continue
            seen.add(key)

            seq += 1
            partida = _normalize_number(it.get("partida"))

            muestra = _normalize_text(it.get("muestra"))
            if muestra:
                m = muestra.lower()
                muestra = "Si" if m in ["si", "sí", "x", "aplica"] else ("No" if m in ["no", "n/a", "na"] else None)

            out.append({
                "partida": partida if partida is not None else seq,
                "subpartida": sub,
                "numero": partida if partida is not None else seq,
                "clave": clave,
                "descripcion": desc,
                "nombre": desc,
                "unidad": _normalize_text(it.get("unidad")) or "PIEZA",
                "cantidad": cant,
                "cantidad_minima": _normalize_number(it.get("cantidad_minima")),
                "cantidad_maxima": _normalize_number(it.get("cantidad_maxima")),
                "presentar_muestra": muestra,
            })

        return {"items_count": len(out), "items": out}

    # ===== Fallback sin OpenAI: heurística básica por fila =====
    rows_text: List[str] = []
    for table in tables:
        rows_text.extend(_matrix_to_rows(_build_table_matrix(table)))
    if not rows_text:
        content = raw_analyze_result.get("content", "") or ""
        rows_text = [l.strip() for l in content.splitlines() if l.strip()]

    fallback = []
    seq = 0
    for line in rows_text:
        cells = [c.strip() for c in line.split("|")]
        non_empty = [c for c in cells if c]
        if len(non_empty) < 2:
            continue
        desc = ""
        for c in non_empty:
            if not re.match(r"^[\d\.,\s/\-]+$", c) and len(c) > len(desc):
                desc = c
        if not desc or len(desc) < 5:
            continue
        cantidad = None
        for c in non_empty:
            num = _normalize_number(c)
            if num is not None and cantidad is None:
                cantidad = num
        seq += 1
        fallback.append({
            "partida": seq,
            "subpartida": None,
            "numero": seq,
            "clave": None,
            "descripcion": desc,
            "nombre": desc,
            "unidad": "PIEZA",
            "cantidad": cantidad,
            "cantidad_minima": None,
            "cantidad_maxima": None,
            "presentar_muestra": None,
        })

    return {"items_count": len(fallback), "items": fallback}