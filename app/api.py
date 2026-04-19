"""
api.py
------
FastAPI backend para WU Matcher.

Endpoints:
  POST /upload-syllabi  – batch: acepta múltiples PDFs
  GET  /health

Ejecutar:
  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
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
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
# Internal: process one PDF → course result dict
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
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload-syllabi")
async def upload_syllabi(files: List[UploadFile] = File(...)):
    """
    Procesa uno o varios PDFs de guías docentes UAM secuencialmente.
    Retorna un listado de cursos, cada uno con sus top matches WU.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Se requiere al menos un fichero PDF.")

    courses = []
    errors = []

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            errors.append({"file": upload.filename, "error": "No es un PDF."})
            continue

        pdf_bytes = await upload.read()
        try:
            course_result = _process_one(pdf_bytes, upload.filename or "")
            courses.append(course_result)
        except Exception as exc:
            errors.append({"file": upload.filename, "error": str(exc)})

    return {
        "status": "success",
        "total": len(courses),
        "courses": courses,
        **({"errors": errors} if errors else {}),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
