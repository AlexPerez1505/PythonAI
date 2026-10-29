import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

from app.services.progress import write_progress

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_CORRECT_CHUNK_ITEMS = int(os.getenv("OPENAI_CORRECT_CHUNK_ITEMS", "120"))
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
OPENAI_CORRECT_MAX_WORKERS = int(os.getenv("OPENAI_CORRECT_MAX_WORKERS", "4"))
OPENAI_CORRECT_ITEMS = os.getenv("OPENAI_CORRECT_ITEMS", "1") == "1"
OPENAI_CORRECT_TIMEOUT = int(os.getenv("OPENAI_CORRECT_TIMEOUT", "45"))
OPENAI_MAX_CANDIDATES_FOR_AI = int(os.getenv("OPENAI_MAX_CANDIDATES_FOR_AI", "240"))
OPENAI_EXTRACT_CHUNK_ROWS = int(os.getenv("OPENAI_EXTRACT_CHUNK_ROWS", "60"))
OPENAI_MAX_ROWS_FOR_AI = int(os.getenv("OPENAI_MAX_ROWS_FOR_AI", "2000"))
OPENAI_STRUCTURE_WITH_AI = os.getenv("OPENAI_STRUCTURE_WITH_AI", "0") == "1"


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
    replacements = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n"}
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _responses_json(system_prompt: str, user_prompt: str, timeout: int = 120) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise Exception("Falta OPENAI_API_KEY en .env")

    url = f"{OPENAI_BASE_URL}/responses"

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "text": {"format": {"type": "json_object"}},
    }

    if OPENAI_MODEL.startswith("gpt-5") and OPENAI_REASONING_EFFORT:
        payload["reasoning"] = {"effort": OPENAI_REASONING_EFFORT}

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

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
    data = _responses_json("Responde únicamente JSON válido. No uses markdown.", prompt)
    return json.dumps(data, ensure_ascii=False)


def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            value = _normalize_text(m.group(1) if m.groups() else m.group(0))
            if value:
                return value[:500]
    return None


def _structure_licitacion_text_fast(raw_text: str) -> Dict[str, Any]:
    """Estructura rápida sin IA. Evita una llamada grande a OpenAI sobre 45k caracteres."""
    text = raw_text or ""
    compact = text[:65000]

    numero = _first_match([
        r"(?:n[uú]mero\s+de\s+procedimiento|procedimiento\s+n[oú\.]*)\s*[:\-]?\s*([^\n\r]{4,160})",
        r"(?:licitaci[oó]n\s+p[uú]blica\s+(?:nacional|internacional)?\s*)\s*[:\-]?\s*([^\n\r]{4,160})",
        r"\b([A-Z]{1,6}[\-/]?\d{2,6}[\-/][A-Z0-9\-/]{2,80})\b",
    ], compact)

    dependencia = _first_match([
        r"(?:dependencia|convocante|entidad convocante|unidad compradora)\s*[:\-]?\s*([^\n\r]{4,220})",
        r"(?:municipio|ayuntamiento)\s+de\s+([^\n\r]{4,180})",
    ], compact)

    objeto = _first_match([
        r"(?:objeto\s+(?:de\s+la\s+)?(?:licitaci[oó]n|contrataci[oó]n)|descripci[oó]n\s+del\s+objeto)\s*[:\-]?\s*([^\n\r]{8,500})",
        r"(?:adquisici[oó]n|contrataci[oó]n|servicio)\s+de\s+([^\n\r]{8,500})",
    ], compact)

    tipo = _first_match([
        r"(licitaci[oó]n\s+p[uú]blica\s+(?:nacional|internacional)[^\n\r]{0,120})",
        r"(invitaci[oó]n\s+a\s+cuando\s+menos\s+tres[^\n\r]{0,120})",
        r"(adjudicaci[oó]n\s+directa[^\n\r]{0,120})",
    ], compact)

    moneda = _first_match([
        r"moneda\s*[:\-]?\s*([^\n\r]{3,80})",
        r"\b(pesos\s+mexicanos|moneda\s+nacional|mxn|usd|d[oó]lares)\b",
    ], compact)

    condiciones_pago = _first_match([
        r"(?:condiciones\s+de\s+pago|forma\s+de\s+pago)\s*[:\-]?\s*([^\n\r]{8,500})",
    ], compact)

    vigencia_contrato = _first_match([
        r"vigencia\s+(?:del\s+)?contrato\s*[:\-]?\s*([^\n\r]{4,300})",
        r"vigencia\s*[:\-]?\s*([^\n\r]{4,300})",
    ], compact)

    lugar_entrega = _first_match([
        r"lugar\s+de\s+entrega\s*[:\-]?\s*([^\n\r]{5,500})",
        r"domicilio\s+de\s+entrega\s*[:\-]?\s*([^\n\r]{5,500})",
    ], compact)

    plazo_entrega = _first_match([
        r"plazo\s+de\s+entrega\s*[:\-]?\s*([^\n\r]{4,300})",
        r"tiempo\s+de\s+entrega\s*[:\-]?\s*([^\n\r]{4,300})",
    ], compact)

    fechas_clave = []
    for label in ["junta de aclaraciones", "presentación de propuestas", "apertura de propuestas", "fallo"]:
        pattern = rf"({label}[^\n\r]{{0,160}}(?:\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}|\d{{1,2}}\s+de\s+[a-záéíóú]+\s+de\s+\d{{4}})[^\n\r]{{0,120}})"
        match = _first_match([pattern], compact)
        if match:
            fechas_clave.append(match)

    anexos = []
    for m in re.finditer(r"\b(anexo\s+[A-Z0-9IVX\-]+[^\n\r]{0,180})", compact, re.IGNORECASE):
        value = _normalize_text(m.group(1))
        if value and value not in anexos:
            anexos.append(value[:250])
        if len(anexos) >= 20:
            break

    return {
        "numero_procedimiento": numero,
        "objeto": objeto,
        "dependencia": dependencia,
        "tipo_procedimiento": tipo,
        "moneda": moneda,
        "anexos": anexos,
        "partidas": [],
        "fechas_clave": fechas_clave,
        "penalizaciones": [],
        "condiciones_pago": condiciones_pago,
        "vigencia_contrato": vigencia_contrato,
        "lugar_entrega": lugar_entrega,
        "plazo_entrega": plazo_entrega,
        "resumen": objeto,
        "fuentes": [],
    }


def structure_licitacion_text(raw_text: str) -> Dict[str, Any]:
    if not OPENAI_STRUCTURE_WITH_AI:
        _log("[structure_licitacion_text] Modo rápido: estructura básica sin OpenAI.")
        return _structure_licitacion_text_fast(raw_text)

    if not OPENAI_API_KEY:
        _log("[structure_licitacion_text] Sin OPENAI_API_KEY. Usando estructura rápida local.")
        return _structure_licitacion_text_fast(raw_text)

    compact_text = (raw_text or "")[:25000]

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


_JUNK_PATTERNS = [
    r"^\d+\s+de\s+\d+$", r"^hoja\s+\d+", r"^p[aá]gina\s+\d+", r"^page\s+\d+",
    r"^fecha$", r"^horario$", r"^domicilio$", r"^lugar$", r"^nombre$",
    r"^firma$", r"^firmas$", r"^subtotal$", r"^total$", r"^iva$", r"^importe$",
    r"^partida$", r"^no\.$", r"^n[uú]mero$", r"^descripci[oó]n$", r"^concepto$",
    r"^cantidad$", r"^unidad$", r"^precio$", r"^precio unitario$",
    r"^costo unitario", r"^costo unitario antes de iva", r"^anexo$", r"^clave$",
    r"^rfc$", r"^tel[eé]fono$", r"^correo$", r"^email$", r"^si$", r"^no$", r"^sí$",
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
        "convocante", "dependencia", "licitacion", "procedimiento",
        "junta de aclaraciones", "presentacion de propuestas", "apertura de propuestas",
        "acto de fallo", "fallo", "contrato", "domicilio", "direccion",
        "servidor publico", "area contratante", "unidad compradora", "compranet",
        "bases", "convocatoria", "aclaraciones", "representante legal", "razon social",
    ]
    if any(word in lower_plain for word in institutional_words) and len(lower_plain) < 160:
        return True
    return False


def _looks_like_product_description(text: str) -> bool:
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
    header_tokens = ["partida", "descripcion", "cantidad", "unidad", "precio unitario", "costo unitario", "importe"]
    if sum(1 for token in header_tokens if token in lower_plain) >= 3:
        return True
    form_tokens = ["fecha", "horario", "domicilio", "lugar", "nombre", "firma", "representante legal"]
    if sum(1 for token in form_tokens if token in lower_plain) >= 2:
        return True
    return False


def _extract_unit_and_clean_description(desc: str, unidad: Optional[str]) -> Tuple[str, str]:
    desc = _normalize_text(desc) or ""
    unidad = _normalize_text(unidad) or ""
    original_desc = desc
    joined_plain = _normalize_for_filter(f"{desc} {unidad}")
    unit_priority = [
        ("paquete", "PAQUETE"), ("paq", "PAQUETE"), ("caja", "CAJA"),
        ("bolsa", "BOLSA"), ("frasco", "FRASCO"), ("botella", "BOTELLA"),
        ("rollo", "ROLLO"), ("juego", "JUEGO"), ("kit", "KIT"), ("servicio", "SERVICIO"),
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
        has_container = any(re.search(rf"\b{re.escape(token)}\b", inside_plain) for token, _ in unit_priority)
        has_piece_count = bool(re.search(r"\b\d+\b", inside_plain)) and bool(re.search(r"\b(pieza|piezas|pza|pz|pzs)\b", inside_plain))
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
        if plain in ["pieza", "piezas", "paquete", "paquetes", "caja", "cajas", "bolsa", "bolsas", "servicio", "servicios"]:
            continue
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
        cantidad_minima if cantidad_minima is not None
        else (cantidad if cantidad is not None
              else (cantidad_maxima if cantidad_maxima is not None else 1))
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
      "partida": null, "subpartida": null, "clave": null,
      "descripcion": "", "unidad": "",
      "cantidad": null, "cantidad_minima": null, "cantidad_maxima": null,
      "cantidad_cotizada": null, "muestra": null
    }
  ]
}
"""


_PROMPT_EXTRACT_TABLE_ROWS = """
Eres un analista de licitaciones de gobierno en México. Recibes renglones de tablas leídas por Azure
Document Intelligence (OCR). Cada renglón viene como celdas separadas por " | " y trae su índice "r".
Tu tarea: devolver TODAS las partidas (bienes o servicios que se solicitan cotizar) que aparezcan en esos renglones.

CÓMO LEER LA TABLA:
- Primero identifica qué columna es qué usando el encabezado. Si el bloque no trae encabezado
  (la tabla continúa en otra página), usa el encabezado de referencia que se te da.
- Las columnas típicas son: partida/número, clave, nombre o descripción del bien, unidad de medida (U.M.),
  cantidad (o mínima/máxima), y observaciones/especificaciones.
- descripcion = el texto de la columna del NOMBRE / DESCRIPCIÓN del bien, aunque sea corto o de una
  sola palabra (BICOLOR, DESENGRAPADORA, CUTTER). NUNCA uses como descripción el texto de observaciones,
  aunque sea más largo.
- especificaciones = el texto de observaciones/especificaciones técnicas de ese mismo renglón (o null).
- El encabezado de la tabla (PARTIDA, NOMBRE, U.M., CANTIDAD ESTIMADA, OBSERVACIONES, etc.) NO es partida.
- Títulos de sección, totales, firmas, fechas, domicilios, notas e instrucciones NO son partidas.

COMPLETITUD (lo más importante):
- Devuelve una partida por cada renglón que tenga un bien o servicio. No omitas ninguna, ni siquiera
  si se parece a otra, se repite o tiene la misma cantidad que otra.
- Si el OCR partió un mismo renglón en dos (por ejemplo la descripción quedó en un renglón y el número,
  unidad y cantidad en el siguiente), únelos en una sola partida.
- No inventes partidas que no estén en los renglones.
- En "r" pon el índice del renglón de donde sale la partida (el del número de partida si se unieron varios).

DESCRIPCIÓN Y UNIDAD:
- Corrige errores leves de OCR sin cambiar el significado del bien.
- Si la descripción trae la presentación entre paréntesis (CAJA CON 50 PIEZAS), quítala de descripcion
  y úsala para la unidad.
- unidad: usa la columna de unidad. Normaliza a PIEZA, PAQUETE, CAJA, BOLSA, FRASCO, BOTELLA, ROLLO,
  JUEGO, KIT, SERVICIO u otra unidad comercial si es la que pide la tabla.

CANTIDAD:
- cantidad = la cantidad solicitada de la partida. No confundas las piezas de la presentación
  (CAJA CON 12 PIEZAS) con la cantidad solicitada.
- Si hay mínima y máxima, llena cantidad_minima y cantidad_maxima.
- cantidad_cotizada = cantidad_minima si existe; si no, cantidad; si no, cantidad_maxima; si no, 1.

PARTIDA:
- partida = el número de partida tal como viene. Si viene 2.15, partida=2 y subpartida=15.
- clave = clave CUCOP/CABMS u otra clave del bien si existe una columna para ello; si no, null.
- muestra = "Si" o "No" solo si hay una columna que lo indique; si no, null.

Responde SOLO JSON válido, sin markdown, con exactamente esta forma:
{
  "items": [
    {
      "r": 0, "partida": null, "subpartida": null, "clave": null,
      "descripcion": "", "especificaciones": null, "unidad": "",
      "cantidad": null, "cantidad_minima": null, "cantidad_maxima": null,
      "cantidad_cotizada": null, "muestra": null
    }
  ]
}
Si en los renglones no hay ninguna partida, devuelve {"items": []}.
"""


def _openai_extract_rows_chunk(header_ref: List[str], rows: List[Dict[str, Any]]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Pide a la IA las partidas de un bloque de renglones. Devuelve (items, None) o (None, error)."""
    user = (
        "Inicio de la primera tabla con estas mismas columnas (normalmente el encabezado; "
        "si son datos, también cuentan solo como referencia, no los devuelvas si no están en los renglones):\n"
        + ("\n".join(header_ref) if header_ref else "(sin encabezado)")
        + "\n\nRenglones:\n"
        + "\n".join(f'r={row["r"]}: {row["text"]}' for row in rows)
    )
    error = None
    for intento in (1, 2):
        try:
            data = _responses_json(_PROMPT_EXTRACT_TABLE_ROWS, user, timeout=OPENAI_CORRECT_TIMEOUT * 3)
            items = data.get("items")
            return (items if isinstance(items, list) else []), None
        except Exception as e:
            error = str(e)[:500]
            _log(f"[extract_rows] ERROR OpenAI con modelo '{OPENAI_MODEL}' (intento {intento}): {error}")
    return None, error


def _table_rows_for_ai(raw_analyze_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Aplana todas las tablas en renglones de texto con índice global y su tabla."""
    rows = []
    for table_index, table in enumerate(raw_analyze_result.get("tables", []) or []):
        for row in _build_table_matrix(table):
            cells = [(c or "").strip() for c in row]
            if not any(cells):
                continue
            rows.append({
                "r": len(rows),
                "table": table_index,
                "cells": cells,
                "text": " | ".join(cells),
            })
    return rows


def _extract_items_with_ai_from_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    La IA lee los renglones de las tablas con su encabezado y decide qué es partida y qué columna es
    la descripción. Si un bloque falla, ese bloque cae al parser local y su error se devuelve.
    """
    chunk_size = max(1, OPENAI_EXTRACT_CHUNK_ROWS)

    # Azure marca como encabezado el primer renglón de cada tabla aunque sea un dato (tablas que siguen
    # en otra página). Como referencia se manda el inicio de la primera tabla con el mismo número de
    # columnas, y la IA decide si es encabezado.
    header_by_cols: Dict[int, List[str]] = {}
    for row in rows:
        cols = len(row["cells"])
        if cols not in header_by_cols:
            header_by_cols[cols] = [r["text"] for r in rows if r["table"] == row["table"]][:2]

    chunks: List[Tuple[List[str], List[Dict[str, Any]]]] = []
    current: List[Dict[str, Any]] = []
    for row in rows:
        if current and (len(current) >= chunk_size or len(current[0]["cells"]) != len(row["cells"])):
            chunks.append((header_by_cols[len(current[0]["cells"])], current))
            current = []
        current.append(row)
    if current:
        chunks.append((header_by_cols[len(current[0]["cells"])], current))

    total = len(chunks)
    _log(f"[extract_rows] {len(rows)} renglones de tabla → {total} bloque(s) para '{OPENAI_MODEL}'")
    write_progress(55, "Leyendo partidas con IA", f"{total} bloque(s) en paralelo")

    results: List[Optional[List[Dict[str, Any]]]] = [None] * total
    errors: List[str] = []
    done = 0
    workers = max(1, min(OPENAI_CORRECT_MAX_WORKERS, total))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_openai_extract_rows_chunk, header, chunk): idx for idx, (header, chunk) in enumerate(chunks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx], error = fut.result()
            except Exception as e:
                error = str(e)[:500]
            if error:
                errors.append(f"Bloque {idx + 1}: {error}")
            done += 1
            write_progress(55 + int(35 * done / total), "Leyendo partidas con IA", f"bloque {done} de {total}")

    items: List[Dict[str, Any]] = []
    for idx, res in enumerate(results):
        if res is None:
            _log(f"[extract_rows] Bloque {idx + 1} sin IA: se usa el parser local para esos renglones.")
            seq = len(items)
            for row in chunks[idx][1]:
                candidate = _row_to_azure_candidate(row["cells"], seq + 1)
                if candidate:
                    seq += 1
                    candidate["_r"] = row["r"]
                    items.append(candidate)
            continue
        for it in res:
            if isinstance(it, dict):
                it["_r"] = it.get("r")
                it["_ai"] = True
                items.append(it)

    items.sort(key=lambda it: it.get("_r") if isinstance(it.get("_r"), int) else 10 ** 9)
    return items, errors


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
        desc = _normalize_text(it.get("descripcion") or it.get("nombre") or it.get("producto") or it.get("description"))
        if not desc:
            continue
        # Lo que la IA ya leyó como partida no pasa por el filtro de palabras: tiraba productos reales.
        if not it.get("_ai") and not _looks_like_product_description(desc):
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
        unidad = _normalize_text(it.get("unidad") or it.get("unit") or it.get("unidad_solicitada")) or "PIEZA"
        if it.get("_ai"):
            # La IA ya tomó la unidad de su columna; inferirla del texto ponía CAJA a "BLOCK VALE DE CAJA".
            unidad = unidad.upper()
        else:
            desc, unidad = _extract_unit_and_clean_description(desc, unidad)
        partida = _normalize_number(it.get("partida"))
        # La partida va en la llave: dos partidas distintas con el mismo texto y cantidad son dos partidas.
        key = (str(partida or ""), (clave or ""), desc.lower()[:180], str(cant or ""), str(cantidad_minima or ""), str(cantidad_maxima or ""), (sub or ""))
        if key in seen:
            continue
        seen.add(key)
        seq += 1
        muestra = _normalize_text(it.get("muestra") or it.get("presentar_muestra"))
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
            "especificaciones": _normalize_text(it.get("especificaciones")),
            "unidad": unidad,
            "cantidad": cant,
            "cantidad_minima": cantidad_minima,
            "cantidad_maxima": cantidad_maxima,
            "cantidad_cotizada": cantidad_cotizada,
            "presentar_muestra": muestra,
        })
    return out


def extract_items_from_azure_tables(raw_analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    tables = raw_analyze_result.get("tables", []) or []

    _log(f"[extract_items] Tablas detectadas por Azure: {len(tables)}")

    rows = _table_rows_for_ai(raw_analyze_result)
    if OPENAI_API_KEY and OPENAI_CORRECT_ITEMS and rows and len(rows) <= OPENAI_MAX_ROWS_FOR_AI:
        write_progress(50, "Leyendo partidas de las tablas", f"{len(rows)} renglones encontrados")
        items, errors = _extract_items_with_ai_from_rows(rows)
        sanitized = _sanitize_extracted_items(items)
        _log(f"[extract_items] Items finales (IA sobre renglones de tabla): {len(sanitized)}")
        write_progress(95, "Limpiando y validando partidas", f"{len(sanitized)} partidas listas")
        # "ia" le dice a Laravel si las partidas vienen de la IA o del parser de respaldo (y por qué).
        return {
            "items_count": len(sanitized),
            "items": sanitized,
            "ia": {"usada": not errors, "modelo": OPENAI_MODEL, "errores": errors},
        }

    if not OPENAI_API_KEY:
        motivo = "El servicio de Python no tiene OPENAI_API_KEY."
    elif not OPENAI_CORRECT_ITEMS:
        motivo = "OPENAI_CORRECT_ITEMS está apagado."
    elif not rows:
        motivo = "Azure no detectó tablas."
    else:
        motivo = f"{len(rows)} renglones superan OPENAI_MAX_ROWS_FOR_AI ({OPENAI_MAX_ROWS_FOR_AI})."
    _log(f"[extract_items] Sin IA para renglones ({motivo}): Azure/parser local extraen candidatos; OpenAI solo corrige.")

    azure_candidates = _extract_candidates_from_azure_tables(raw_analyze_result)
    _log(f"[extract_items] Candidatos extraídos desde Azure/parser local: {len(azure_candidates)}")
    write_progress(50, "Leyendo partidas de las tablas", f"{len(azure_candidates)} renglones encontrados")

    if not azure_candidates:
        return {"items_count": 0, "items": []}

    corrected_all: List[Dict[str, Any]] = []

    use_openai_correction = bool(OPENAI_API_KEY and OPENAI_CORRECT_ITEMS and len(azure_candidates) <= OPENAI_MAX_CANDIDATES_FOR_AI)

    if use_openai_correction:
        chunk_size = max(1, OPENAI_CORRECT_CHUNK_ITEMS)
        chunks = [azure_candidates[i:i + chunk_size] for i in range(0, len(azure_candidates), chunk_size)]
        total_bloques = len(chunks)

        _log(f"[correct_items] Corrigiendo {len(azure_candidates)} candidatos con '{OPENAI_MODEL}' "
             f"(reasoning={OPENAI_REASONING_EFFORT}) en {total_bloques} bloque(s) EN PARALELO")
        write_progress(55, "Revisando partidas con IA", f"{total_bloques} bloque(s) en paralelo")

        # Corrección EN PARALELO: todos los bloques al mismo tiempo (más rápido).
        results_by_index: List[Optional[List[Dict[str, Any]]]] = [None] * total_bloques
        done = 0
        workers = max(1, min(OPENAI_CORRECT_MAX_WORKERS, total_bloques))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_openai_correct_items_chunk, ch): idx for idx, ch in enumerate(chunks)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    _log(f"[correct_items] Bloque {idx + 1} falló: {e}")
                    res = None
                results_by_index[idx] = res if res else chunks[idx]
                done += 1
                _log(f"[correct_items] Bloque {done}/{total_bloques} listo")
                write_progress(55 + int(35 * done / total_bloques), "Revisando partidas con IA", f"bloque {done} de {total_bloques}")

        for res in results_by_index:
            corrected_all.extend(res or [])
    else:
        if OPENAI_API_KEY and OPENAI_CORRECT_ITEMS and len(azure_candidates) > OPENAI_MAX_CANDIDATES_FOR_AI:
            _log(f"[correct_items] Se omite OpenAI por velocidad: {len(azure_candidates)} candidatos > límite {OPENAI_MAX_CANDIDATES_FOR_AI}.")
            write_progress(88, "Corrección rápida local", f"{len(azure_candidates)} candidatos")
        else:
            _log("[correct_items] Corrección rápida local sin OpenAI.")
        corrected_all = azure_candidates

    sanitized = _sanitize_extracted_items(corrected_all)
    _log(f"[extract_items] Items finales después de corrección/sanitizado: {len(sanitized)}")
    write_progress(95, "Limpiando y validando partidas", f"{len(sanitized)} partidas listas")

    return {
        "items_count": len(sanitized),
        "items": sanitized,
        "ia": {"usada": False, "modelo": OPENAI_MODEL, "errores": [motivo]},
    }


def debug_hash_bytes(file_bytes: bytes, label: str = "PDF") -> None:
    _log(f"[{label}] sha256: {hashlib.sha256(file_bytes).hexdigest()}")
    _log(f"[{label}] size bytes: {len(file_bytes)}")