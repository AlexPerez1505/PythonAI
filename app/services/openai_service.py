import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4")

client = OpenAI(api_key=OPENAI_API_KEY)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


# ===================== FILTROS ANTI-BASURA =====================

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


def _looks_like_junk(text: str) -> bool:
    normalized = _normalize_text(text) or ""
    lower = normalized.lower()
    lower_plain = _normalize_for_filter(normalized)

    if len(lower_plain) < 4:
        return True

    for pattern in _JUNK_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return True

    for pattern in _JUNK_PATTERNS:
        if re.search(pattern, lower_plain, re.IGNORECASE):
            return True

    # Puro número, fecha, horario o folio corto.
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
    normalized = _normalize_text(text) or ""

    if _looks_like_junk(normalized):
        return False

    letters = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", normalized)

    if len(letters) < 8:
        return False

    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", normalized)

    if len(words) < 2:
        return False

    lower_plain = _normalize_for_filter(normalized)

    product_signals = [
        "papel",
        "cartulina",
        "folder",
        "folders",
        "pluma",
        "lapiz",
        "lapices",
        "boligrafo",
        "marcador",
        "corrector",
        "pegamento",
        "cinta",
        "tijera",
        "grapadora",
        "grapas",
        "clip",
        "clips",
        "broche",
        "arillo",
        "block",
        "notas",
        "sobre",
        "sobres",
        "carpeta",
        "archivero",
        "toner",
        "tinta",
        "cartucho",
        "bateria",
        "caja",
        "paquete",
        "pieza",
        "piezas",
        "servicio",
        "mantenimiento",
        "instalacion",
        "suministro",
        "equipo",
        "material",
    ]

    # No exigimos que tenga señal de producto, pero si la tiene, pasa con más confianza.
    if any(signal in lower_plain for signal in product_signals):
        return True

    # Si es una frase larga con letras suficientes, puede ser descripción técnica.
    if len(words) >= 4 and len(normalized) >= 20:
        return True

    return False


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


# ===================== EXTRACCIÓN CON OPENAI =====================

_PROMPT_SISTEMA_ITEMS = """
Eres un extractor experto de partidas de anexos de licitaciones públicas de gobierno en México.
Tu trabajo es interpretar tablas de licitación con sentido común comercial y devolver solamente productos o servicios reales que se puedan cotizar.
Te paso filas de una tabla; cada fila trae sus celdas separadas por ' | '.

REGLAS GENERALES:
- Devuelve SOLO JSON válido.
- No uses markdown.
- No inventes productos.
- Omite encabezados, pies de página, fechas, horarios, domicilios, firmas, notas, instrucciones, subtotales, totales y números de página.
- Si una fila no describe un bien o servicio concreto, omítela.
- No devuelvas como producto textos como FECHA, HORARIO, DOMICILIO, LUGAR, NOMBRE, FIRMA, 1 de 16, Hoja 3, Descripción, Cantidad, Unidad, Precio, etc.

PARTIDA Y SUBPARTIDA:
- Si la partida viene como 1, devuelve partida=1 y subpartida=null.
- Si viene como 1.1, devuelve partida=1 y subpartida=1.
- Si viene como 2.15, devuelve partida=2 y subpartida=15.
- Si hay columna de partida y columna de subpartida, respétalas.
- No confundas cantidad con partida.

DESCRIPCIÓN:
- descripcion debe ser SOLO el producto o servicio real solicitado.
- No pongas en descripcion la unidad, cantidad, clave, precio ni encabezados.
- Si la descripción trae una presentación entre paréntesis, sepárala cuando sea unidad/presentación.
- Ejemplo: 'Arillo metálico doble p engargolar 3/8" (caja/paquete 90 piezas)' debe quedar:
  descripcion='Arillo metálico doble p engargolar 3/8"'
  unidad='PAQUETE'
- Ejemplo: 'Marcador permanente negro (caja con 12 piezas)' debe quedar:
  descripcion='Marcador permanente negro'
  unidad='CAJA'
- Ejemplo: 'Bolígrafo punto mediano (paquete con 10 piezas)' debe quedar:
  descripcion='Bolígrafo punto mediano'
  unidad='PAQUETE'
- Si la descripción dice algo como 'Block con 100 notas autoadheribles' y eso es el nombre normal del producto, conserva esa frase en descripcion.

UNIDAD DE MEDIDA:
- Usa sentido común comercial.
- unidad debe ser la unidad comercial SIMPLE con la que se compra/cotiza: PAQUETE, CAJA, BOLSA, FRASCO, BOTELLA, ROLLO, JUEGO, KIT, SERVICIO, PIEZA, etc.
- Si el texto dice 'caja/paquete 90 piezas', usa unidad='PAQUETE' o unidad='CAJA', pero NUNCA unidad='PIEZA'.
- Si el texto dice 'paquete con 100 piezas', usa unidad='PAQUETE', no 'PIEZA'.
- Si el texto dice 'caja con 12 piezas', usa unidad='CAJA', no 'PIEZA'.
- Si el texto dice 'bolsa con 50 piezas', usa unidad='BOLSA', no 'PIEZA'.
- Si el texto dice solo 'pieza' y no hay otra presentación, usa unidad='PIEZA'.
- No metas la presentación completa en unidad; no pongas 'PAQUETE CON 100 PIEZAS'. Pon solamente 'PAQUETE'.
- No dejes la presentación dentro de descripcion cuando sea claramente unidad/presentación.
- El número de piezas de una presentación no es la cantidad solicitada. Ejemplo: cantidad=10 y unidad='PAQUETE' con presentación de 90 piezas significa que solicitan 10 paquetes.

CANTIDADES:
- cantidad_minima debe ser la cantidad mínima si aparece.
- cantidad_maxima debe ser la cantidad máxima si aparece.
- cantidad debe ser la cantidad única si solo hay una cantidad.
- cantidad_cotizada debe seguir esta regla obligatoria:
  1. Si existe cantidad_minima, cantidad_cotizada = cantidad_minima.
  2. Si no existe cantidad_minima pero existe cantidad, cantidad_cotizada = cantidad.
  3. Si no existe cantidad ni cantidad_minima pero existe cantidad_maxima, cantidad_cotizada = cantidad_maxima.
  4. Si no existe ninguna cantidad, cantidad_cotizada = 1.
- Nunca uses cantidad_maxima como cantidad_cotizada si también existe cantidad_minima.
- No confundas números de partida, claves, páginas o presentaciones con cantidad solicitada.
- En 'PAQUETE CON 100 PIEZAS', el 100 pertenece a la presentación, no necesariamente a cantidad solicitada.

CLAVE:
- clave es código, clave presupuestal, SKU, clave CABMS o código de partida si aparece.
- Si no existe, usa null.

Devuelve exactamente este JSON:
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
        # IMPORTANTE: stderr, nunca stdout, porque Laravel puede leer stdout como JSON.
        _log(f"[extract_items] ERROR OpenAI con modelo '{OPENAI_MODEL}': {e}")
        return []



def _extract_unit_and_clean_description(desc: str, unidad: Optional[str]) -> tuple[str, str]:
    """
    Usa sentido común mínimo como red de seguridad si la IA deja la presentación
    dentro de la descripción o devuelve PIEZA cuando realmente era CAJA/PAQUETE/etc.
    """
    desc = _normalize_text(desc) or ""
    unidad = _normalize_text(unidad) or ""

    original_desc = desc
    joined = f"{desc} {unidad}".lower()
    joined_plain = _normalize_for_filter(joined)

    # Preferencia cuando aparecen varias palabras. En "caja/paquete 90 piezas"
    # normalmente cotizas por paquete/presentación, y sobre todo no por PIEZA.
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

    # Si la unidad venía vacía o era PIEZA pero hay presentación en el texto,
    # reemplazamos por la unidad comercial inferida.
    unidad_plain = _normalize_for_filter(unidad)
    if inferred and (not unidad or unidad_plain in ["pieza", "piezas", "pza", "pz", "pzs"]):
        unidad = inferred

    # Si la unidad venía compuesta, simplificarla: "PAQUETE CON 100 PIEZAS" -> "PAQUETE"
    if not inferred and unidad:
        unidad_text_plain = _normalize_for_filter(unidad)
        for token, unit_name in unit_priority:
            if re.search(rf"\b{re.escape(token)}\b", unidad_text_plain):
                unidad = unit_name
                break

    # Quitar paréntesis de presentación de la descripción:
    # "(caja/paquete 90 piezas)", "(caja con 12 piezas)", "(paquete c/10 piezas)".
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

        unidad = _normalize_text(it.get("unidad")) or "PIEZA"
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

        muestra = _normalize_text(it.get("muestra"))
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
    tables = raw_analyze_result.get("tables", []) or []
    api_key = os.getenv("OPENAI_API_KEY", "")

    _log(f"[extract_items] Tablas detectadas por Azure: {len(tables)}")

    crudos_all: List[Dict[str, Any]] = []

    if api_key and tables:
        for table_index, table in enumerate(tables, start=1):
            rows = _matrix_to_rows(_build_table_matrix(table))

            if len(rows) < 2:
                continue

            useful_rows = []

            for row in rows:
                if _row_looks_like_header_or_form(row):
                    continue

                # No quitamos filas cortas aquí porque OpenAI puede necesitar contexto,
                # pero sí quitamos basura obvia de página.
                row_plain = _normalize_for_filter(row)
                if re.match(r"^\d+\s+de\s+\d+$", row_plain):
                    continue

                useful_rows.append(row)

            if not useful_rows:
                continue

            _log(f"[extract_items] Tabla {table_index}: {len(useful_rows)} filas útiles candidatas")

            # Extrae por lotes de 45 filas.
            for i in range(0, len(useful_rows), 45):
                chunk = useful_rows[i:i + 45]
                crudos_all.extend(_openai_extract_chunk(chunk))

    sanitized = _sanitize_extracted_items(crudos_all)

    if sanitized:
        _log(f"[extract_items] Items válidos extraídos con OpenAI: {len(sanitized)}")
        return {
            "items_count": len(sanitized),
            "items": sanitized,
        }

    # ===================== FALLBACK SEGURO SIN OPENAI =====================
    # Este fallback ahora es conservador. Es mejor devolver pocos items correctos
    # que meter FECHA, 1 de 16, DOMICILIO, etc.

    _log("[extract_items] OpenAI no devolvió items válidos. Entrando a fallback seguro.")

    rows_text: List[str] = []

    for table in tables:
        rows_text.extend(_matrix_to_rows(_build_table_matrix(table)))

    if not rows_text:
        content = raw_analyze_result.get("content", "") or ""
        rows_text = [line.strip() for line in content.splitlines() if line.strip()]

    fallback = []
    seq = 0
    seen = set()

    unit_words = [
        "PIEZA",
        "PIEZAS",
        "PZA",
        "PZ",
        "PZS",
        "CAJA",
        "CAJAS",
        "PAQUETE",
        "PAQUETES",
        "PAQ",
        "SERVICIO",
        "SERVICIOS",
        "KG",
        "KGS",
        "GRAMO",
        "GRAMOS",
        "LT",
        "LTS",
        "LITRO",
        "LITROS",
        "ML",
        "METRO",
        "METROS",
        "M",
        "CM",
        "MM",
        "ROLLO",
        "ROLLOS",
        "BOLSA",
        "BOLSAS",
        "BOTELLA",
        "BOTELLAS",
    ]

    for line in rows_text:
        if _row_looks_like_header_or_form(line):
            continue

        cells = [c.strip() for c in line.split("|")]
        non_empty = [c for c in cells if c]

        if len(non_empty) < 2:
            continue

        candidate_descriptions = []

        for c in non_empty:
            if _looks_like_product_description(c):
                candidate_descriptions.append(c)

        if not candidate_descriptions:
            continue

        # Elegimos la celda descriptiva más larga.
        desc = max(candidate_descriptions, key=len)
        desc = _normalize_text(desc)

        if not desc or not _looks_like_product_description(desc):
            continue

        desc_key = desc.lower()[:180]
        if desc_key in seen:
            continue

        seen.add(desc_key)

        cantidad = None
        unidad = None

        for c in non_empty:
            clean = c.strip()
            clean_plain = _normalize_for_filter(clean)

            if re.match(r"^\d+\s+de\s+\d+$", clean_plain):
                continue

            if clean.upper() in unit_words:
                unidad = clean.upper()
                continue

            # Si la celda es un número pequeño y está antes de la descripción,
            # normalmente es número de partida, no cantidad.
            num = _normalize_number(clean)
            if num is not None and num > 0:
                # Evita tomar 1,2,3... como cantidad cuando parecen índice.
                if num <= 300:
                    cantidad = num
                else:
                    cantidad = num

        desc, unidad = _extract_unit_and_clean_description(desc, unidad)

        seq += 1

        fallback.append({
            "partida": seq,
            "subpartida": None,
            "numero": seq,
            "clave": None,
            "descripcion": desc,
            "nombre": desc,
            "unidad": unidad or "PIEZA",
            "cantidad": cantidad,
            "cantidad_minima": None,
            "cantidad_maxima": None,
            "cantidad_cotizada": cantidad if cantidad is not None else 1,
            "presentar_muestra": None,
        })

    _log(f"[extract_items] Items válidos extraídos con fallback: {len(fallback)}")

    return {
        "items_count": len(fallback),
        "items": fallback,
    }


def debug_hash_bytes(file_bytes: bytes, label: str = "PDF") -> None:
    """
    Úsalo desde el script que llama a Azure para confirmar si realmente
    estás mandando un PDF nuevo o el mismo archivo anterior.
    """
    _log(f"[{label}] sha256: {hashlib.sha256(file_bytes).hexdigest()}")
    _log(f"[{label}] size bytes: {len(file_bytes)}")