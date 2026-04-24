import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.services.azure_service import analyze_pdf_with_auto_split
from app.services.openai_service import structure_licitacion_text, extract_items_from_azure_tables

app = FastAPI(title="Python AI Service")

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"
UPLOADS_DIR = STORAGE_DIR / "uploads"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def job_file(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def save_job(job_id: str, payload: dict) -> None:
    final_path = job_file(job_id)
    temp_path = JOBS_DIR / f"{job_id}.tmp"

    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(final_path)


def load_job(job_id: str, retries: int = 5, delay: float = 0.15) -> dict | None:
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

        job["status"] = "completed"
        job["result"] = {
            "ok": True,
            "status": result.get("status"),
            "content_preview": full_content[:2000],
            "full_text": full_content,
            "pages_count": len(analyze_result.get("pages", [])),
            "tables_count": len(analyze_result.get("tables", [])),
            "raw_analyze_result": analyze_result,  # IMPORTANTE
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


@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Python AI service running",
    }


@app.post("/documents/analyze-async")
async def analyze_document_async(
    file: UploadFile = File(...),
    pages_per_chunk: int = Form(5),
):
    try:
        job_id = str(uuid.uuid4())
        ext = Path(file.filename or "document.pdf").suffix or ".pdf"
        upload_path = UPLOADS_DIR / f"{job_id}{ext}"

        content = await file.read()
        upload_path.write_bytes(content)

        job_payload = {
            "job_id": job_id,
            "status": "queued",
            "filename": file.filename,
            "pages_per_chunk": pages_per_chunk,
        }
        save_job(job_id, job_payload)

        thread = threading.Thread(
            target=process_document_job,
            args=(job_id, str(upload_path), pages_per_chunk),
            daemon=True,
        )
        thread.start()

        return {
            "ok": True,
            "job_id": job_id,
            "status": "queued",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": str(e)},
        )


@app.get("/documents/jobs/{job_id}")
def get_job_status(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"},
        )

    if not job:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "message": "Job no encontrado"},
        )

    return {
        "ok": True,
        "job_id": job_id,
        "status": job.get("status"),
        "error": job.get("error"),
    }


@app.get("/documents/jobs/{job_id}/result")
def get_job_result(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"},
        )

    if not job:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "message": "Job no encontrado"},
        )

    if job.get("status") != "completed":
        return {
            "ok": True,
            "job_id": job_id,
            "status": job.get("status"),
            "error": job.get("error"),
        }

    return {
        "ok": True,
        "job_id": job_id,
        "status": "completed",
        "result": job.get("result"),
    }


@app.get("/documents/jobs/{job_id}/structured")
def get_structured_result(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"},
        )

    if not job:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "message": "Job no encontrado"},
        )

    if job.get("status") != "completed":
        return {
            "ok": True,
            "job_id": job_id,
            "status": job.get("status"),
            "error": job.get("error"),
        }

    result = job.get("result") or {}
    full_text = result.get("full_text", "")

    if not full_text:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": "El job no tiene texto OCR disponible."},
        )

    try:
        structured = structure_licitacion_text(full_text)

        return {
            "ok": True,
            "job_id": job_id,
            "status": "completed",
            "structured": structured,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": f"Error estructurando con OpenAI: {str(e)}"},
        )


@app.get("/documents/jobs/{job_id}/items")
def get_items_result(job_id: str):
    try:
        job = load_job(job_id)
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"ok": False, "message": f"Job temporalmente no disponible: {str(e)}"},
        )

    if not job:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "message": "Job no encontrado"},
        )

    if job.get("status") != "completed":
        return {
            "ok": True,
            "job_id": job_id,
            "status": job.get("status"),
            "error": job.get("error"),
        }

    result = job.get("result") or {}
    raw_analyze_result = result.get("raw_analyze_result")

    if not raw_analyze_result:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "message": "El job no tiene raw_analyze_result disponible."},
        )

    try:
        items = extract_items_from_azure_tables(raw_analyze_result)

        return {
            "ok": True,
            "job_id": job_id,
            "status": "completed",
            "items_result": items,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "message": f"Error extrayendo partidas desde tablas: {str(e)}"},
        )