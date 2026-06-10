import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_CORRECT_CHUNK_ITEMS = int(os.getenv("OPENAI_CORRECT_CHUNK_ITEMS", "80"))


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


def _normalize_for_filter(text: str) -> str:
    text = _normalize_text(text) or ""
    text = text.lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "ñ": "n",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# OpenAI Responses API
# ============================================================

def _responses_json(system_prompt: str, user_prompt: str, timeout: int = 180) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise Exception("Falta OPENAI_API_KEY en .env")

    url = f"{OPENAI_BASE_URL}/responses"

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "text": {
            "format": {
                "type": "json_object",
            }
        },
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)

    if not response.ok:
        raise Exception(f"OpenAI Responses API error: {response.status_code} - {response.text}")

    data = response.json()

    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        raw_text = data["output_text"]
    else:
        parts = []

        for output in data.get("output", []) or []:
            for content in output.get("content", []) or []:
                if isinstance(content, dict):
                    if content.get("type") in ["output_text", "text"] and content.get("text"):
                        parts.append(content.get("text"))
                    elif content.get("text"):
                        parts.append(content.get("text"))

        raw_text = "\n".join(parts).strip()

    if not raw_text:
        raise Exception(f"OpenAI no devolvió texto utilizable: {data}")

    return _safe_json(raw_text)


def _openai_text(prompt: str) -> str:
    data = _responses_json(
        "Responde únicamente JSON válido. No uses markdown.",
        prompt,
    )
    return json.dumps(data, ensure_ascii=False)


# ============================================================
# Resumen estructurado de la licitación
# ============================================================

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
- No inventes partidas.
- No uses encabezados, fechas, páginas, domicilios, firmas o instrucciones como productos.

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
    except Exception as e:
        _log(f"[structure_licitacion_text] Error estructurando texto: {e}")
        return {"partidas": []}


# ============================================================
# Lectura de tablas Azure
# ============================================================

def _build_table_matrix(table: Dict[str, Any]) -> List[List[str]]:
    row_count = table.get("rowCount", 0)
    col_count = table.get("columnCount", 0)

    matrix = [["" for _ in range(col_count)] for _ in range(row_count)]

    for cell in table.get("cells", []) or []:
        r = cell.get("rowIndex", 0)
        c = cell.get("columnIndex", 0)
        content = _normalize_text(cell.get("content", "")) or ""

        row_span = int(cell.get("rowSpan") or 1)
        col_span = int(cell.get("columnSpan") or 1)

        for rr in range(r, min(r + row_span, row_count)):
            for cc in range(c, min(c + col_span, col_count)):
                if 0 <= rr < row_count and 0 <= cc < col_count:
                    if not matrix[rr][cc]:
                        matrix[rr][cc] = content

    return matrix


def _matrix_to_rows(matrix: List[List[str]]) -> List[str]:
    rows = []

    for row in matrix:
        cells = [(c or "").strip() for c in row]

        if any(cells):
            rows.append(" | ".join(cells))

    return rows


# ============================================================
# Filtros de basura: solo eliminan basura obvia
# ============================================================

_JUNK_PATTERNS = [
    r"^\d+\s+de\s+\d+$",
    r"^hoja\s+\d+",
    r"^p[aá]gina\s+\d+",
    r"^page\s+\d+",
    r"^fecha$",
    r"^horario$",
    r"^domicilio$",
    r"^lugar$",
    r"^nombre$",
    r"^firma$",
    r"^firmas$",
    r"^subtotal$",
    r"^total$",
    r"^iva$",
    r"^importe$",
    r"^partida$",
    r"^no\.$",
    r"^n[uú]mero$",
    r"^descripci[oó]n$",
    r"^concepto$",
    r"^cantidad$",
    r"^unidad$",
    r"^precio$",
    r"^precio unitario$",
    r"^costo unitario",
    r"^costo unitario antes de iva",
    r"^anexo$",
    r"^clave$",
    r"^rfc$",
    r"^tel[eé]fono$",
    r"^correo$",
    r"^email$",
    r"^si$",
    r"^no$",
    r"^sí$",
]


def _looks_like_junk(text: str) -> bool:
    normalized = _normalize_text(text) or ""
    lower = normalized.lower()
    lower_plain = _normalize_for_filter(normalized)

    if len(lower_plain) < 2:
        return True

    for pattern in _JUNK_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True

    for pattern in _JUNK_PATTERNS:
        if re.search(pattern, lower_plain, re.IGNORECASE):
            return True

    if re.match(r"^[\d\s.,/\-:]+$", lower_plain):
        return True

    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$", lower_plain):
        return True

    if re.match(r"^\d{1,2}:\d{2}", lower_plain):
        return True

    if re.search(r"\b\d{1,2}:\d{2}\s*a\s*\d{1,2}:\d{2}\b", lower_plain):
        return True

    institutional_words = [
        "convocante",
        "dependencia",
        "licitacion",
        "procedimiento",
        "junta de aclaraciones",
        "presentacion de propuestas",
        "apertura de propuestas",
        "acto de fallo",
        "fallo",
        "contrato",
        "domicilio",
        "direccion",
        "servidor publico",
        "area contratante",
        "unidad compradora",
        "compranet",
        "bases",
        "convocatoria",
        "aclaraciones",
        "representante legal",
        "razon social",
    ]

    if any(word in lower_plain for word in institutional_words) and len(lower_plain) < 160:
        return True

    return False


def _looks_like_product_description(text: str) -> bool:
    """
    Este filtro NO debe ser estricto.
    Azure ya leyó la tabla. Aquí solo evitamos basura evidente.
    Productos reales pueden ser de una palabra: Servilletas, Aguja, Grapas, Clips.
    """
    normalized = _normalize_text(text) or ""

    if _looks_like_junk(normalized):
        return False

    letters = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", normalized)

    if len(letters) < 4:
        return False

    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", normalized)

    if not words:
        return False

    if len(words) == 1:
        return len(words[0]) >= 4

    return True


def _row_looks_like_header_or_form(row_text: str) -> bool:
    lower_plain = _normalize_for_filter(row_text)

    header_tokens = [
        "partida",
        "descripcion",
        "cantidad",
        "unidad",
        "precio unitario",
        "costo unitario",
        "importe",
    ]

    hits = sum(1 for token in header_tokens if token in lower_plain)

    if hits >= 3:
        return True

    form_tokens = [
        "fecha",
        "horario",
        "domicilio",
        "lugar",
        "nombre",
        "firma",
        "representante legal",
    ]

    form_hits = sum(1 for token in form_tokens if token in lower_plain)

    if form_hits >= 2:
        return True

    return False


# ============================================================
# Limpieza de unidad/descripcion
# ============================================================

def _extract_unit_and_clean_description(desc: str, unidad: Optional[str]) -> Tuple[str, str]:
    desc = _normalize_text(desc) or ""
    unidad = _normalize_text(unidad) or ""

    original_desc = desc
    joined_plain = _normalize_for_filter(f"{desc} {unidad}")

    unit_priority = [
        ("paquete", "PAQUETE"),
        ("paq", "PAQUETE"),
        ("caja", "CAJA"),
        ("bolsa", "BOLSA"),
        ("frasco", "FRASCO"),
        ("botella", "BOTELLA"),
        ("rollo", "ROLLO"),
        ("juego", "JUEGO"),
        ("kit", "KIT"),
        ("servicio", "SERVICIO"),
    ]

    inferred = None

    for token, unit_name in unit_priority:
        if re.search(rf"\b{re.escape(token)}\b", joined_plain):
            inferred = unit_name
            break

    unidad_plain = _normalize_for_filter(unidad)

    if inferred and (not unidad or unidad_plain in ["pieza", "piezas", "pza", "pz", "pzs"]):
        unidad = inferred

    if not inferred and unidad:
        unidad_text_plain = _normalize_for_filter(unidad)

        for token, unit_name in unit_priority:
            if re.search(rf"\b{re.escape(token)}\b", unidad_text_plain):
                unidad = unit_name
                break

    def should_remove_parenthetical(match: re.Match) -> str:
        inside = match.group(1)
        inside_plain = _normalize_for_filter(inside)
        has_container = any(
            re.search(rf"\b{re.escape(token)}\b", inside_plain)
            for token, _ in unit_priority
        )
        has_piece_count = bool(re.search(r"\b\d+\b", inside_plain)) and bool(
            re.search(r"\b(pieza|piezas|pza|pz|pzs)\b", inside_plain)
        )

        if has_container or has_piece_count:
            return ""

        return match.group(0)

    desc = re.sub(r"\(([^)]{1,120})\)", should_remove_parenthetical, desc)
    desc = _normalize_text(desc) or original_desc

    if not unidad:
        unidad = "PIEZA"

    return desc, unidad.upper()


def _parse_partida_subpartida(value) -> Tuple[Optional[int], Optional[int]]:
    txt = _normalize_text(value)

    if not txt:
        return None, None

    if re.match(r"^\d+\.\d+$", txt):
        a, b = txt.split(".", 1)
        return int(a), int(b)

    num = _normalize_number(txt)

    if isinstance(num, int):
        return num, None

    if isinstance(num, float) and num.is_integer():
        return int(num), None

    return None, None


def _detect_unit_from_text(text: str) -> Optional[str]:
    plain = _normalize_for_filter(text)

    if "paquete" in plain or re.search(r"\bpaq\b", plain):
        return "PAQUETE"

    if "caja" in plain:
        return "CAJA"

    if "bolsa" in plain:
        return "BOLSA"

    if "frasco" in plain:
        return "FRASCO"

    if "botella" in plain:
        return "BOTELLA"

    if "rollo" in plain:
        return "ROLLO"

    if "juego" in plain:
        return "JUEGO"

    if "servicio" in plain:
        return "SERVICIO"

    if plain in ["pieza", "piezas", "pza", "pz", "pzs"]:
        return "PIEZA"

    return None


# ============================================================
# Azure extrae candidatos. OpenAI solo corrige candidatos.
# ============================================================

def _row_to_azure_candidate(cells: List[str], seq: int) -> Optional[Dict[str, Any]]:
    non_empty = [_normalize_text(c) or "" for c in cells if (_normalize_text(c) or "")]

    if len(non_empty) < 2:
        return None

    joined = " | ".join(non_empty)

    if _row_looks_like_header_or_form(joined):
        return None

    partida = None
    subpartida = None

    for c in non_empty[:3]:
        p, s = _parse_partida_subpartida(c)

        if p is not None:
            partida = p
            subpartida = s
            break

    numbers = []

    for index, c in enumerate(non_empty):
        plain = _normalize_for_filter(c)

        if re.match(r"^\d+\s+de\s+\d+$", plain):
            continue

        p, _s = _parse_partida_subpartida(c)

        if p is not None and p == partida and index <= 2:
            continue

        n = _normalize_number(c)

        if n is not None and n > 0:
            numbers.append(n)

    cantidad = numbers[0] if numbers else None
    cantidad_minima = None
    cantidad_maxima = None

    if len(numbers) >= 2:
        cantidad_minima = numbers[0]
        cantidad_maxima = numbers[1]
        cantidad = None

    desc_candidates = []

    for c in non_empty:
        if not _looks_like_product_description(c):
            continue

        plain = _normalize_for_filter(c)

        if plain in [
            "pieza",
            "piezas",
            "paquete",
            "paquetes",
            "caja",
            "cajas",
            "bolsa",
            "bolsas",
            "servicio",
            "servicios",
        ]:
            continue

        # Evita que números/cantidades ganen como descripción.
        if _normalize_number(c) is not None:
            continue

        desc_candidates.append(c)

    if not desc_candidates:
        return None

    descripcion = max(desc_candidates, key=len)

    unidad = None

    for c in non_empty:
        detected = _detect_unit_from_text(c)

        if detected:
            unidad = detected
            break

    descripcion, unidad = _extract_unit_and_clean_description(descripcion, unidad)

    cantidad_cotizada = (
        cantidad_minima
        if cantidad_minima is not None
        else (
            cantidad
            if cantidad is not None
            else (
                cantidad_maxima
                if cantidad_maxima is not None
                else 1
            )
        )
    )

    return {
        "partida": partida if partida is not None else seq,
        "subpartida": subpartida,
        "numero": partida if partida is not None else seq,
        "clave": None,
        "descripcion": descripcion,
        "nombre": descripcion,
        "unidad": unidad or "PIEZA",
        "cantidad": cantidad,
        "cantidad_minima": cantidad_minima,
        "cantidad_maxima": cantidad_maxima,
        "cantidad_cotizada": cantidad_cotizada,
        "presentar_muestra": None,
    }


def _extract_candidates_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = raw_analyze_result.get("tables", []) or []

    candidates = []
    seen = set()
    seq = 0

    for table_index, table in enumerate(tables, start=1):
        matrix = _build_table_matrix(table)

        for row in matrix:
            cells = [(c or "").strip() for c in row]

            if not any(cells):
                continue

            candidate = _row_to_azure_candidate(cells, seq + 1)

            if not candidate:
                continue

            key = (
                str(candidate.get("partida") or ""),
                str(candidate.get("subpartida") or ""),
                (candidate.get("descripcion") or "").lower()[:180],
                str(candidate.get("cantidad") or ""),
                str(candidate.get("cantidad_minima") or ""),
                str(candidate.get("cantidad_maxima") or ""),
            )

            if key in seen:
                continue

            seen.add(key)
            seq += 1

            if not candidate.get("partida"):
                candidate["partida"] = seq
                candidate["numero"] = seq

            candidates.append(candidate)

    return candidates


_PROMPT_CORRECT_ITEMS = """
Eres un revisor de partidas ya extraídas por Azure Document Intelligence.
IMPORTANTE: Azure ya hizo la extracción. Tú NO debes buscar partidas nuevas en el PDF ni inventar renglones.
Tu tarea es únicamente corregir, limpiar y validar los candidatos que ya vienen en JSON.

REGLAS:
- Devuelve SOLO JSON válido.
- No uses markdown.
- No inventes partidas.
- No agregues partidas nuevas.
- No elimines productos reales aunque sean simples o de una sola palabra, por ejemplo:
  Aguja, Aguja para alacrán, Servilletas, Borrador para pizarrón, Etiquetas para CD, Liga de hule, Marcatextos, Clips, Grapas.
- Solo elimina renglones que claramente NO son producto/servicio:
  FECHA, HORARIO, DOMICILIO, LUGAR, NOMBRE, FIRMA, 1 de 16, Hoja 3, Página 5,
  encabezados como Descripción, Cantidad, Unidad, Precio, Subtotal, Total, IVA,
  notas, instrucciones, direcciones, dependencias, firmas o cláusulas.

DESCRIPCIÓN:
- Corrige errores de OCR u ortografía leves.
- descripcion debe ser solo el producto o servicio.
- Si la descripción trae presentación entre paréntesis, quítala de descripcion y úsala para unidad.
- Ejemplo:
  descripcion entrada: Arillo metálico doble p engargolar 3/8" (caja/paquete 90 piezas)
  descripcion salida: Arillo metálico doble p engargolar 3/8"
  unidad salida: PAQUETE
- No cambies el significado del producto.

UNIDAD:
- Corrige la unidad si Azure la puso mal.
- Si el texto dice caja/paquete, paquete con piezas, caja con piezas o bolsa con piezas, NO uses PIEZA.
- Usa unidad simple comercial: PAQUETE, CAJA, BOLSA, FRASCO, BOTELLA, ROLLO, JUEGO, KIT, SERVICIO, PIEZA.
- Si dice caja/paquete 90 piezas, usa PAQUETE.
- Si dice paquete con 100 piezas, usa PAQUETE.
- Si dice caja con 12 piezas, usa CAJA.
- Si dice bolsa con 50 piezas, usa BOLSA.
- Si solo dice pieza y no hay otra presentación, usa PIEZA.

CANTIDAD:
- No confundas piezas de presentación con cantidad solicitada.
- Si existe cantidad_minima, cantidad_cotizada debe ser cantidad_minima.
- Si no existe cantidad_minima pero existe cantidad, cantidad_cotizada debe ser cantidad.
- Si no existe cantidad ni cantidad_minima pero existe cantidad_maxima, cantidad_cotizada debe ser cantidad_maxima.
- Si no existe ninguna cantidad, cantidad_cotizada debe ser 1.

PARTIDA Y SUBPARTIDA:
- Respeta partida y subpartida de Azure.
- Si partida viene como 1.1, salida partida=1 y subpartida=1.
- Si viene 2.15, salida partida=2 y subpartida=15.

Devuelve exactamente:
{
  "items": [
    {
      "partida": null,
      "subpartida": null,
      "clave": null,
      "descripcion": "",
      "unidad": "",
      "cantidad": null,
      "cantidad_minima": null,
      "cantidad_maxima": null,
      "cantidad_cotizada": null,
      "muestra": null
    }
  ]
}
"""


def _openai_correct_items_chunk(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        data = _responses_json(
            _PROMPT_CORRECT_ITEMS,
            "Corrige y valida estos candidatos extraídos por Azure. "
            "No inventes ni agregues otros. JSON:\n\n"
            + json.dumps({"items": items}, ensure_ascii=False),
        )

        corrected = data.get("items") or []

        return corrected if isinstance(corrected, list) else []

    except Exception as e:
        _log(f"[correct_items] ERROR OpenAI con modelo '{OPENAI_MODEL}': {e}")
        return []


def _sanitize_extracted_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    seq = 0

    for it in items:
        if not isinstance(it, dict):
            continue

        desc = _normalize_text(
            it.get("descripcion")
            or it.get("nombre")
            or it.get("producto")
            or it.get("description")
        )

        if not desc:
            continue

        if not _looks_like_product_description(desc):
            _log(f"[extract_items] Omitiendo basura/no-producto: {desc}")
            continue

        clave = _normalize_text(it.get("clave"))
        sub = _normalize_text(it.get("subpartida"))

        cant = _normalize_number(it.get("cantidad"))
        cantidad_minima = _normalize_number(it.get("cantidad_minima"))
        cantidad_maxima = _normalize_number(it.get("cantidad_maxima"))
        cantidad_cotizada = _normalize_number(it.get("cantidad_cotizada"))

        if cantidad_cotizada is None:
            if cantidad_minima is not None:
                cantidad_cotizada = cantidad_minima
            elif cant is not None:
                cantidad_cotizada = cant
            elif cantidad_maxima is not None:
                cantidad_cotizada = cantidad_maxima
            else:
                cantidad_cotizada = 1

        unidad = _normalize_text(
            it.get("unidad")
            or it.get("unit")
            or it.get("unidad_solicitada")
        ) or "PIEZA"

        desc, unidad = _extract_unit_and_clean_description(desc, unidad)

        key = (
            (clave or ""),
            desc.lower()[:180],
            str(cant or ""),
            str(cantidad_minima or ""),
            str(cantidad_maxima or ""),
            (sub or ""),
        )

        if key in seen:
            continue

        seen.add(key)
        seq += 1

        partida = _normalize_number(it.get("partida"))

        muestra = _normalize_text(it.get("muestra") or it.get("presentar_muestra"))

        if muestra:
            m = muestra.lower()

            if m in ["si", "sí", "x", "aplica"]:
                muestra = "Si"
            elif m in ["no", "n/a", "na"]:
                muestra = "No"
            else:
                muestra = None

        out.append({
            "partida": partida if partida is not None else seq,
            "subpartida": sub,
            "numero": partida if partida is not None else seq,
            "clave": clave,
            "descripcion": desc,
            "nombre": desc,
            "unidad": unidad,
            "cantidad": cant,
            "cantidad_minima": cantidad_minima,
            "cantidad_maxima": cantidad_maxima,
            "cantidad_cotizada": cantidad_cotizada,
            "presentar_muestra": muestra,
        })

    return out


def extract_items_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flujo correcto:
    1. Azure Document Intelligence extrae tablas.
    2. Parser local convierte tablas de Azure en candidatos.
    3. OpenAI solo corrige OCR/unidad/ortografía y elimina basura obvia.
    """
    tables = raw_analyze_result.get("tables", []) or []

    _log(f"[extract_items] Tablas detectadas por Azure: {len(tables)}")
    _log("[extract_items] Azure/parser local extraen candidatos; OpenAI solo corrige.")

    azure_candidates = _extract_candidates_from_azure_tables(raw_analyze_result)

    _log(f"[extract_items] Candidatos extraídos desde Azure/parser local: {len(azure_candidates)}")

    if not azure_candidates:
        return {
            "items_count": 0,
            "items": [],
        }

    corrected_all: List[Dict[str, Any]] = []

    if OPENAI_API_KEY:
        chunk_size = max(1, OPENAI_CORRECT_CHUNK_ITEMS)
        _log(f"[correct_items] Corrigiendo candidatos con OpenAI en bloques de {chunk_size}")

        for i in range(0, len(azure_candidates), chunk_size):
            chunk = azure_candidates[i:i + chunk_size]
            _log(f"[correct_items] Bloque {int(i / chunk_size) + 1}: {len(chunk)} items")
            corrected = _openai_correct_items_chunk(chunk)

            if corrected:
                corrected_all.extend(corrected)
            else:
                corrected_all.extend(chunk)
    else:
        corrected_all = azure_candidates

    sanitized = _sanitize_extracted_items(corrected_all)

    _log(f"[extract_items] Items finales después de corrección/sanitizado: {len(sanitized)}")

    return {
        "items_count": len(sanitized),
        "items": sanitized,
    }


def debug_hash_bytes(file_bytes: bytes, label: str = "PDF") -> None:
    _log(f"[{label}] sha256: {hashlib.sha256(file_bytes).hexdigest()}")
    _log(f"[{label}] size bytes: {len(file_bytes)}")
