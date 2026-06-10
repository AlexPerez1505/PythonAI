import json
import os
from pathlib import Path


def write_progress(pct, etapa, detalle=""):
    """Escribe el avance REAL del análisis en el archivo que indique AI_PROGRESS_FILE."""
    path = os.getenv("AI_PROGRESS_FILE", "")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {"pct": int(pct), "etapa": str(etapa), "detalle": str(detalle)},
                f,
                ensure_ascii=False,
            )
        os.replace(tmp, path)
    except Exception:
        pass