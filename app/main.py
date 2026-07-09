import os
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
BASE_DIR = APP_DIR.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
import json
import threading
import time
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.services.azure_service import analyze_pdf_with_auto_split
from app.services.openai_service import extract_items_from_azure_tables
from app.services.progress import write_progress

app = FastAPI(title="Python AI Service")

STORAGE_DIR = BASE_DIR / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
UPLOADS_DIR = STORAGE_DIR / "uploads"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def save_job(job_id: str, payload: Dict[str, Any]) -> None:
    final_path = job_file(job_id)
    temp_path = JOBS_DIR / f"{job_id}.tmp"
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(final_path)


def load_job(job_id: str, retries: int = 5, delay: float = 0.15) -> Optional[Dict[str, Any]]:
    path = job_file(job_id)
    if not path.exists():
        return None

    last_error = None
    for _ in range(retries):
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if not raw:
                raise ValueError("Archivo de job vacío temporalmente.")
            return json.loads(raw)
        except Exception as e:
            last_error = e
            time.sleep(delay)

    raise last_error


def process_document_job(job_id: str, file_path: str, pages_per_chunk: int) -> None:
    try:
        job = load_job(job_id) or {}
        job["status"] = "processing"
        save_job(job_id, job)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        result = analyze_pdf_with_auto_split(
            file_bytes=file_bytes,
            model="prebuilt-layout",
            pages_per_chunk=pages_per_chunk,
        )

        analyze_result = result.get("analyzeResult", {})
        full_content = analyze_result.get("content", "") or ""

        items = extract_items_from_azure_tables(analyze_result)

        job["status"] = "completed"
        job["result"] = {
            "ok": True,
            "status": result.get("status"),
            "content_preview": full_content[:2000],
            "pages_count": len(analyze_result.get("pages", [])),
            "tables_count": len(analyze_result.get("tables", [])),
            "items_json": items,
        }
        save_job(job_id, job)

    except Exception as e:
        try:
            job = load_job(job_id) or {}
        except Exception:
            job = {}

        job["status"] = "failed"
        job["error"] = str(e)
        save_job(job_id, job)


def analyze_document_cli(file_path: str, run_id: str, pages_per_chunk: int, filename: str) -> Dict[str, Any]:
    write_progress(10, "Leyendo documento", filename)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # Azure escribe su propio progreso interno (15% -> 40%) mientras lee el PDF.
    result = analyze_pdf_with_auto_split(
        file_bytes=file_bytes,
        model="prebuilt-layout",
        pages_per_chunk=pages_per_chunk,
    )

    analyze_result = result.get("analyzeResult", {})
    full_content = analyze_result.get("content", "") or ""

    write_progress(45, "Documento leído", "Extrayendo partidas...")

    try:
        # extract_items escribe su propio progreso interno (50% -> 95%).
        items = extract_items_from_azure_tables(analyze_result)
    except Exception as e:
        items = {
            "ok": False,
            "error": f"Error extrayendo partidas/productos desde el documento: {str(e)}",
            "items_count": 0,
            "items": [],
        }

    write_progress(97, "Partidas listas", "Preparando resultado...")

    return {
        "ok": True,
        "status": "completed",
        "run_id": run_id,
        "filename": filename,
        "result_json": {
            "ok": True,
            "status": result.get("status"),
            "content_preview": full_content[:2000],
            "pages_count": len(analyze_result.get("pages", [])),
            "tables_count": len(analyze_result.get("tables", [])),
        },
        "structured_json": {
            "ok": True,
            "message": "Omitido. Solo se extraen productos/servicios solicitados para cotización.",
        },
        "items_json": items,
    }


@app.get("/")
def home():
    return {"ok": True, "message": "Python AI service running"}


@app.post("/documents/analyze-async")
async def analyze_document_async(file: UploadFile = File(...), pages_per_chunk: int = Form(5)):
    try:
        job_id = str(uuid.uuid4())
        ext = Path(file.filename or "document.pdf").suffix or ".pdf"
        upload_path = UPLOADS_DIR / f"{job_id}{ext}"

        content = await file.read()
        upload_path.write_bytes(content)

        save_job(job_id, {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "pages_per_chunk": pages_per_chunk,
        })

        thread = threading.Thread(
            target=process_document_job,
            args=(job_id, str(upload_path), pages_per_chunk),
            daemon=True,
        )
        thread.start()

        return {"ok": True, "job_id": job_id, "status": "queued"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "message": str(e)})


@app.get("/documents/jobs/{job_id}")
def get_job_status(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(status_code=503, content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"})

    if not job:
        return JSONResponse(status_code=404, content={"ok": False, "message": "Job no encontrado"})

    return {"ok": True, "job_id": job_id, "status": job.get("status"), "error": job.get("error")}


@app.get("/documents/jobs/{job_id}/result")
def get_job_result(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(status_code=503, content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"})

    if not job:
        return JSONResponse(status_code=404, content={"ok": False, "message": "Job no encontrado"})

    if job.get("status") != "completed":
        return {"ok": True, "job_id": job_id, "status": job.get("status"), "error": job.get("error")}

    return {"ok": True, "job_id": job_id, "status": "completed", "result": job.get("result")}


@app.get("/documents/jobs/{job_id}/items")
def get_items_result(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(status_code=503, content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"})

    if not job:
        return JSONResponse(status_code=404, content={"ok": False, "message": "Job no encontrado"})

    if job.get("status") != "completed":
        return {"ok": True, "job_id": job_id, "status": job.get("status"), "error": job.get("error")}

    result = job.get("result") or {}

    return {
        "ok": True,
        "job_id": job_id,
        "status": "completed",
        "items_result": result.get("items_json") or {"items_count": 0, "items": []},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python AI CLI runner")
    parser.add_argument("--file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pages-per-chunk", type=int, default=5)
    parser.add_argument("--filename", default="document.pdf")
    parser.add_argument("--progress-file", default="")

    args = parser.parse_args()

    # progress.py lee AI_PROGRESS_FILE del entorno. Lo seteamos desde el argumento
    # para que funcione tanto si PHP lo pasó por env como si solo lo pasó por --progress-file.
    if args.progress_file:
        os.environ["AI_PROGRESS_FILE"] = args.progress_file

    try:
        write_progress(5, "Iniciando análisis", args.filename)

        output = analyze_document_cli(
            file_path=args.file,
            run_id=args.run_id,
            pages_per_chunk=args.pages_per_chunk,
            filename=args.filename,
        )

        write_progress(100, "Análisis completado", "Generando cotización...")

        print(json.dumps(output, ensure_ascii=False))
        sys.exit(0)

    except Exception as e:
        write_progress(100, "Error", str(e))
        print(str(e), file=sys.stderr)
        sys.exit(1)