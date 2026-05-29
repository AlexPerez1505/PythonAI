import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional

try:
    from azure_client import analyze_pdf_with_auto_split
except Exception as e:  # pragma: no cover
    analyze_pdf_with_auto_split = None
    _IMPORT_ERROR = e


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    t = re.sub(r"\s+", " ", str(v)).strip()
    return t or None


def _money(v) -> Optional[float]:
    if v is None:
        return None
    s = re.sub(r"[^\d.,]", "", str(v))
    if not s:
        return None
    # quitar separador de miles (coma) dejando el punto decimal
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", "")
    try:
        return round(float(s), 2)
    except Exception:
        return None


def _matrix(table: Dict[str, Any]) -> List[List[str]]:
    rows = table.get("rowCount", 0)
    cols = table.get("columnCount", 0)
    m = [["" for _ in range(cols)] for _ in range(rows)]
    for cell in table.get("cells", []) or []:
        r = cell.get("rowIndex", 0)
        c = cell.get("columnIndex", 0)
        if 0 <= r < rows and 0 <= c < cols:
            m[r][c] = _norm(cell.get("content", "")) or ""
    return m


KIND_KEYWORDS = {
    "numero": ["partida", "núm", "num", "no.", "no ", "renglon", "renglón", "concepto no"],
    "descripcion": ["descrip", "concepto", "bien", "servicio", "producto", "articulo", "artículo"],
    "cantidad": ["cantidad", "cant"],
    "empresa": ["licitante", "empresa", "proveedor", "razon social", "razón social",
                "oferente", "participante", "concursante", "postor"],
    "precio": ["importe", "precio", "monto", "total", "unitario", "económic", "economic", "oferta"],
    "gano": ["adjudic", "ganad", "gana", "fallo", "resultado", "asignad"],
}


def _kind_of(header: str) -> Optional[str]:
    t = (header or "").lower()
    for kind, kws in KIND_KEYWORDS.items():
        if any(k in t for k in kws):
            return kind
    return None


COMPANY_HINTS = ["s.a", "sa de", "s. de r", "s de rl", "s.a.p.i", "sapi",
                 "comercializadora", "distribuidora", "grupo", "proveedora",
                 "soluciones", "corporativo", "s.c", "de c.v"]


def _looks_company(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 4:
        return False
    low = t.lower()
    if any(h in low for h in COMPANY_HINTS):
        return True
    # encabezado en mayúsculas con varias palabras = probable empresa
    letters = re.sub(r"[^A-Za-zÁÉÍÓÚÑ ]", "", t)
    return len(letters.split()) >= 2 and t.upper() == t


def _is_jureto(empresa: str) -> bool:
    return "jureto" in (empresa or "").lower()


def _truthy_win(val: str) -> bool:
    t = (val or "").strip().lower()
    return t in ["si", "sí", "x", "1", "adjudicado", "ganador", "gana", "asignado", "✓"]


def _header_row(matrix: List[List[str]]) -> Optional[int]:
    best_i, best_score = None, 0
    for i, row in enumerate(matrix[:6]):
        score = sum(1 for c in row if _kind_of(c))
        score += sum(1 for c in row if _looks_company(c))
        if score > best_score:
            best_score, best_i = score, i
    return best_i if best_score >= 2 else None


def _parse_long(matrix: List[List[str]], hidx: int) -> List[Dict[str, Any]]:
    """Formato largo: una fila por (empresa) con columnas partida/empresa/precio/gano."""
    header = matrix[hidx]
    cmap: Dict[str, int] = {}
    for idx, h in enumerate(header):
        k = _kind_of(h)
        if k and k not in cmap:
            cmap[k] = idx

    if "empresa" not in cmap or "precio" not in cmap:
        return []

    partidas: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for row in matrix[hidx + 1:]:
        def g(key):
            i = cmap.get(key)
            return _norm(row[i]) if (i is not None and i < len(row)) else None

        empresa = g("empresa")
        precio = _money(g("precio"))
        if not empresa and precio is None:
            continue

        label = g("numero") or "1"
        desc = g("descripcion")
        key = str(label)

        if key not in partidas:
            partidas[key] = {
                "partida_label": label,
                "descripcion": desc,
                "cantidad": _money(g("cantidad")),
                "ofertas": [],
            }
            order.append(key)

        if desc and not partidas[key]["descripcion"]:
            partidas[key]["descripcion"] = desc

        gano = False
        if "gano" in cmap:
            gano = _truthy_win(g("gano") or "")

        if empresa:
            partidas[key]["ofertas"].append({
                "empresa": empresa,
                "precio": precio,
                "es_jureto": _is_jureto(empresa),
                "gano": gano,
            })

    return [partidas[k] for k in order]


def _parse_wide(matrix: List[List[str]], hidx: int) -> List[Dict[str, Any]]:
    """Formato ancho: cada columna de empresa tiene su precio por partida."""
    header = matrix[hidx]
    num_idx = desc_idx = None
    company_cols: List[int] = []

    for idx, h in enumerate(header):
        k = _kind_of(h)
        if k == "numero" and num_idx is None:
            num_idx = idx
        elif k == "descripcion" and desc_idx is None:
            desc_idx = idx
        elif _looks_company(h):
            company_cols.append(idx)

    if not company_cols or (num_idx is None and desc_idx is None):
        return []

    out = []
    for r, row in enumerate(matrix[hidx + 1:], start=1):
        label = _norm(row[num_idx]) if (num_idx is not None and num_idx < len(row)) else str(r)
        desc = _norm(row[desc_idx]) if (desc_idx is not None and desc_idx < len(row)) else None

        ofertas = []
        for ci in company_cols:
            if ci >= len(row):
                continue
            precio = _money(row[ci])
            if precio is None:
                continue
            empresa = header[ci]
            ofertas.append({
                "empresa": empresa,
                "precio": precio,
                "es_jureto": _is_jureto(empresa),
                "gano": False,
            })

        if not ofertas:
            continue

        out.append({
            "partida_label": label,
            "descripcion": desc,
            "cantidad": None,
            "ofertas": ofertas,
        })

    return out


def _decide_winner(partida: Dict[str, Any]) -> None:
    ofertas = partida.get("ofertas", [])
    if not ofertas:
        return
    # si nadie viene marcado como ganador, gana el menor precio (criterio común en gobierno)
    if not any(o.get("gano") for o in ofertas):
        con_precio = [o for o in ofertas if o.get("precio") is not None]
        if con_precio:
            menor = min(con_precio, key=lambda o: o["precio"])
            menor["gano"] = True


def extract_fallo_from_azure(analyze_result: Dict[str, Any]) -> Dict[str, Any]:
    tables = analyze_result.get("tables", []) or []
    content = analyze_result.get("content", "") or ""

    partidas: List[Dict[str, Any]] = []

    for table in tables:
        matrix = _matrix(table)
        hidx = _header_row(matrix)
        if hidx is None:
            continue

        parsed = _parse_long(matrix, hidx)
        if not parsed:
            parsed = _parse_wide(matrix, hidx)

        partidas.extend(parsed)

    for p in partidas:
        _decide_winner(p)

    # metadatos sencillos del texto
    numero_acta = None
    m = re.search(r"(LA|EA|IA|AA)[-\s][\w\-/]+", content)
    if m:
        numero_acta = m.group(0).strip()

    fecha_fallo = None
    md = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", content)
    if md:
        d, mo, y = md.group(1), md.group(2), md.group(3)
        fecha_fallo = f"{y}-{int(mo):02d}-{int(d):02d}"

    return {
        "ok": True,
        "numero_acta": numero_acta,
        "fecha_fallo": fecha_fallo,
        "partidas": partidas,
        "content": content[:60000],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--pages-per-chunk", type=int, default=5)
    args = parser.parse_args()

    if analyze_pdf_with_auto_split is None:
        print(json.dumps({"ok": False, "message": f"No se pudo importar azure_client: {_IMPORT_ERROR}"}))
        sys.exit(1)

    try:
        with open(args.file, "rb") as f:
            data = f.read()

        result = analyze_pdf_with_auto_split(data, model="prebuilt-layout", pages_per_chunk=args.pages_per_chunk)
        analyze_result = result.get("analyzeResult", result)

        payload = extract_fallo_from_azure(analyze_result)
        print(json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"ok": False, "message": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()