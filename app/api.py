"""
api.py
------
FastAPI backend para WU Matcher.

Endpoints:
  POST /upload-syllabi  – lanza procesamiento en background, retorna job_id
  GET  /job/{job_id}    – polling: cursos procesados hasta ahora
  GET  /search-wu       – búsqueda de asignaturas WU por nombre
  GET  /health

Ejecutar:
  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from scraper.parse_my_syllabus import parse_pdf
from pipeline.retrieval import HybridRetriever
from rag.generator import (
    generate_justification,
    _extract_section_text,
    _course_entry_by_code,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WU Matcher API",
    description="Encuentra equivalencias entre asignaturas UAM y WU Vienna",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory job state
# ---------------------------------------------------------------------------

job_results: dict[str, list] = {}   # job_id -> list of processed course dicts
job_status: dict[str, dict] = {}    # job_id -> {total, completed, done, errors}

# ---------------------------------------------------------------------------
# Lazy singleton retriever
# ---------------------------------------------------------------------------

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Internal: process one PDF → course result dict (synchronous, CPU-bound)
# ---------------------------------------------------------------------------

def _process_one(pdf_bytes: bytes, filename: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        query_course = parse_pdf(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if query_course.get("parse_error"):
        raise ValueError(f"No se pudo parsear '{filename}': ¿es un plan docente UAM?")

    query_course.setdefault("source", "uam")

    retriever = _get_retriever()
    top5_raw = retriever.process_query_course(query_course, top_courses=5)

    chunks_path = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    top5_enriched = []
    for match in top5_raw:
        code = match.get("code", "")
        entry = _course_entry_by_code(chunks_data, code)
        top5_enriched.append({
            **match,
            "contents": _extract_section_text(entry, "contents") if entry else "",
            "learning_outcomes": _extract_section_text(entry, "learning_outcomes") if entry else "",
        })

    justification = generate_justification(query_course, top5_enriched)

    llm_by_code: dict[str, dict] = {
        m.get("code", ""): m for m in justification.get("matches", [])
    }

    matches = []
    for r in top5_raw:
        code = r.get("code", "")
        llm = llm_by_code.get(code, {})
        matches.append({
            "rank": r["rank"],
            "code": code,
            "name": r.get("name", ""),
            "credits": r.get("credits", ""),
            "type": r.get("type", ""),
            "match_percentage": llm.get("match_percentage"),
            "recommendation": llm.get("recommendation", ""),
            "overlapping_topics": llm.get("overlapping_topics", []),
            "missing_in_wu": llm.get("missing_in_wu", []),
            "extra_in_wu": llm.get("extra_in_wu", []),
            "schedule": r.get("schedule", []),
        })

    return {
        "code": query_course.get("code", ""),
        "name": query_course.get("name", ""),
        "credits": query_course.get("credits", ""),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Background task: process all PDFs for a job
# ---------------------------------------------------------------------------

async def process_all(job_id: str, files_data: list[tuple[bytes, str]]) -> None:
    for pdf_bytes, filename in files_data:
        try:
            # Run the blocking call in a thread so the event loop stays free
            course_result = await asyncio.to_thread(_process_one, pdf_bytes, filename)
            job_results[job_id].append(course_result)
        except Exception as exc:
            job_status[job_id]["errors"].append({"file": filename, "error": str(exc)})
        finally:
            job_status[job_id]["completed"] += 1
    job_status[job_id]["done"] = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload-syllabi")
async def upload_syllabi(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Lee los PDFs, lanza el procesamiento en background y retorna un job_id
    inmediatamente. El cliente debe hacer polling a GET /job/{job_id}.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Se requiere al menos un fichero PDF.")

    files_data: list[tuple[bytes, str]] = []
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = await upload.read()
        files_data.append((pdf_bytes, upload.filename))

    if not files_data:
        raise HTTPException(status_code=400, detail="No se encontraron PDFs válidos.")

    job_id = str(uuid.uuid4())
    job_results[job_id] = []
    job_status[job_id] = {
        "total": len(files_data),
        "completed": 0,
        "done": False,
        "errors": [],
    }

    background_tasks.add_task(process_all, job_id, files_data)
    return {"job_id": job_id}


@app.get("/job/{job_id}")
def get_job(job_id: str):
    """Polling endpoint: devuelve el estado actual del job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job no encontrado.")

    status = job_status[job_id]
    return {
        "status": "done" if status["done"] else "processing",
        "total": status["total"],
        "completed": status["completed"],
        "courses": list(job_results[job_id]),
        "errors": status["errors"],
    }


@app.get("/search-wu")
def search_wu(q: str = "", limit: int = 10):
    """Búsqueda de asignaturas WU por nombre. Sin query devuelve las primeras 20."""
    chunks_path = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
    chunks_data: list = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not q.strip():
        subset = chunks_data[:20]
    else:
        q_lower = q.strip().lower()
        # Exact code match comes first, then name/code partial matches
        matches_exact = [c for c in chunks_data if c["code"] == q.strip()]
        matches_rest = [
            c for c in chunks_data
            if c not in matches_exact
            and (q_lower in c["name"].lower() or q_lower in c["code"].lower())
        ]
        subset = (matches_exact + matches_rest)[:limit]

    return [
        {
            "code": c["code"],
            "name": c["name"],
            "credits": c["metadata"].get("credits", ""),
            "type": c["metadata"].get("type", ""),
            "schedule": c["metadata"].get("schedule", []),
        }
        for c in subset
    ]


@app.get("/health")
def health():
    return {"status": "ok"}
