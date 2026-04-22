"""
api.py
------
FastAPI backend para WU Matcher.

Endpoints:
  POST /upload-syllabi  – procesa PDFs y retorna resultado completo
  GET  /search-wu       – búsqueda de asignaturas WU por nombre o código
  GET  /health

Ejecutar:
  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path
from typing import List
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scraper.parse_my_syllabus import parse_pdf
from pipeline.retrieval import HybridRetriever
from rag.generator import (
    generate_justification,
    _extract_section_text,
    _course_entry_by_code,
    _get_client as _get_llm_client,
    MODEL_NAME as LLM_MODEL,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WU Matcher API",
    description="Encuentra equivalencias entre asignaturas UAM y WU Vienna",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Optativas: áreas temáticas válidas
# ---------------------------------------------------------------------------

AREAS_VALID = {
    "MARKETING", "FINANZAS", "DATA_SCIENCE", "GESTIÓN",
    "ESTRATEGIA", "DERECHO", "TECNOLOGÍA", "SOSTENIBILIDAD", "RECURSOS_HUMANOS",
}


def _classify_areas(name: str, contents: str) -> list[str]:
    """Llama al LLM para clasificar una asignatura en áreas temáticas."""
    client = _get_llm_client()
    trunc = (contents or "")[:800]
    prompt = (
        f"Asignatura: {name}\n\nContenidos:\n{trunc}\n\n"
        f"Clasifica esta asignatura en una o más de estas categorías:\n"
        f"MARKETING, FINANZAS, DATA_SCIENCE, GESTIÓN, ESTRATEGIA, "
        f"DERECHO, TECNOLOGÍA, SOSTENIBILIDAD, RECURSOS_HUMANOS.\n"
        f"Devuelve SOLO los nombres separados por comas, sin explicación ni texto adicional."
    )
    response = client.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.message.content.strip()
    areas = [a.strip().upper() for a in raw.replace("\n", ",").split(",")]
    valid = [a for a in areas if a in AREAS_VALID]
    return valid if valid else ["GESTIÓN"]


# ---------------------------------------------------------------------------
# Pydantic models para /match-optatives
# ---------------------------------------------------------------------------

class OptativeItem(BaseModel):
    code: str
    name: str
    areas_tematicas: list[str] = []


class MatchOptativesRequest(BaseModel):
    optatives: list[OptativeItem]
    user_interests: str


# ---------------------------------------------------------------------------
# In-memory job store for async processing
# ---------------------------------------------------------------------------

job_results: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Lazy singleton retriever (se inicializa una sola vez al primer uso)
# ---------------------------------------------------------------------------

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Pipeline: procesa un PDF y retorna el dict de resultado
# ---------------------------------------------------------------------------

def _process_one(pdf_bytes: bytes, filename: str) -> dict:
    """
    Orquesta el pipeline completo para un PDF:
      1. parse_pdf         → extrae code, name, credits, contents
      2. process_query_course → top 5 matches WU (hybrid retrieval)
      3. generate_justification → match_percentage, recommendation, topics
    """
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

    # ── Retrieval ─────────────────────────────────────────────────────────
    retriever = _get_retriever()
    top5_raw = retriever.process_query_course(query_course, top_courses=5)

    # ── Enriquecer con contents/outcomes para el generador ────────────────
    chunks_path = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    top5_enriched = []
    for match in top5_raw:
        code = match.get("code", "")
        entry = _course_entry_by_code(chunks_data, code)
        top5_enriched.append({
            **match,
            "contents":          _extract_section_text(entry, "contents")          if entry else "",
            "learning_outcomes": _extract_section_text(entry, "learning_outcomes") if entry else "",
        })

    # ── Generación (LLM) ──────────────────────────────────────────────────
    justification = generate_justification(query_course, top5_enriched)
    llm_by_code: dict[str, dict] = {
        m.get("code", ""): m for m in justification.get("matches", [])
    }

    # ── Combinar retrieval + LLM ──────────────────────────────────────────
    matches = []
    for r in top5_raw:
        code = r.get("code", "")
        llm  = llm_by_code.get(code, {})
        matches.append({
            "rank":               r["rank"],
            "code":               code,
            "name":               r.get("name", ""),
            "credits":            r.get("credits", ""),
            "type":               r.get("type", ""),
            "match_percentage":   llm.get("match_percentage"),
            "recommendation":     llm.get("recommendation", ""),
            "overlapping_topics": llm.get("overlapping_topics", []),
            "missing_in_wu":      llm.get("missing_in_wu", []),
            "extra_in_wu":        llm.get("extra_in_wu", []),
            "schedule":           r.get("schedule", []),
        })

    return {
        "code":    query_course.get("code", ""),
        "name":    query_course.get("name", ""),
        "credits": query_course.get("credits", ""),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/upload-syllabi")
async def upload_syllabi(files: List[UploadFile] = File(...)):
    """
    Procesa uno o varios PDFs de guías docentes UAM.
    Espera a que todos terminen y retorna la respuesta completa.
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

    courses: list[dict] = []
    errors:  list[dict] = []

    for pdf_bytes, filename in files_data:
        try:
            # asyncio.to_thread evita bloquear el event loop durante CPU/IO intensivo
            course_result = await asyncio.to_thread(_process_one, pdf_bytes, filename)
            courses.append(course_result)
        except Exception as exc:
            errors.append({"file": filename, "error": str(exc)})

    return {
        "status":  "success",
        "total":   len(files_data),
        "courses": courses,
        "errors":  errors,
    }


async def _process_all_async(job_id: uuid.UUID, files_data: list[tuple[bytes, str]]) -> None:
    for pdf_bytes, filename in files_data:
        try:
            course_result = await asyncio.to_thread(_process_one, pdf_bytes, filename)
            job_results[job_id]["courses"].append(course_result)
        except Exception as exc:
            job_results[job_id]["errors"].append({"file": filename, "error": str(exc)})
    job_results[job_id]["status"] = "done"


@app.post("/upload-syllabi-async")
async def upload_syllabi_async(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Inicia el procesamiento en background y retorna un job_id para polling."""
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

    job_id = uuid.uuid4()
    job_results[job_id] = {"courses": [], "errors": [], "status": "processing", "total": len(files_data)}
    background_tasks.add_task(_process_all_async, job_id, files_data)
    return {"job_id": str(job_id)}


@app.get("/job/{job_id}")
def get_job(job_id: uuid.UUID):
    job = job_results.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job no encontrado.")
    courses = job["courses"]
    return {
        "status":    job["status"],
        "courses":   courses,
        "total":     job["total"],
        "completed": len(courses),
    }


@app.post("/upload-optatives")
async def upload_optatives(files: List[UploadFile] = File(...)):
    """
    Procesa PDFs de optativas UAM: extrae info via parse_pdf y clasifica
    en áreas temáticas via LLM. Guarda en data/my_courses/{code}_optative.json.
    """
    if not files:
        raise HTTPException(status_code=400, detail="Se requiere al menos un fichero PDF.")

    my_courses_dir = _PROJECT_ROOT / "data" / "my_courses"
    my_courses_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    errors:  list[dict] = []

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = await upload.read()
        filename  = upload.filename

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = Path(tmp.name)
            try:
                query_course = await asyncio.to_thread(parse_pdf, tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            if query_course.get("parse_error"):
                raise ValueError(f"No se pudo parsear '{filename}'")

            code     = query_course.get("code", "")
            name     = query_course.get("name", "")
            credits  = query_course.get("credits", "")
            contents = query_course.get("contents", "")

            areas = await asyncio.to_thread(_classify_areas, name, contents)

            optative_data = {
                "code": code, "name": name, "credits": credits,
                "contents": contents, "areas_tematicas": areas,
                "source": "uam_optativa",
            }
            json_path = my_courses_dir / f"{code}_optative.json"
            json_path.write_text(
                json.dumps(optative_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            results.append({"code": code, "name": name, "credits": credits, "areas_tematicas": areas})
        except Exception as exc:
            errors.append({"file": filename, "error": str(exc)})

    return {"optatives": results, "errors": errors}


@app.post("/match-optatives")
async def match_optatives(body: MatchOptativesRequest):
    """
    Para cada optativa UAM rankea asignaturas WU por similitud semántica
    combinando los contenidos de la optativa con los intereses del usuario.
    """
    retriever      = _get_retriever()
    my_courses_dir = _PROJECT_ROOT / "data" / "my_courses"
    matched: list[dict] = []

    for opt in body.optatives:
        json_path = my_courses_dir / f"{opt.code}_optative.json"
        if json_path.exists():
            opt_data = json.loads(json_path.read_text(encoding="utf-8"))
            contents = opt_data.get("contents", "")
        else:
            contents = opt.name

        areas_str = ", ".join(opt.areas_tematicas) if opt.areas_tematicas else ""
        parts: list[str] = []
        if contents:
            parts.append(contents[:600])
        if areas_str:
            parts.append(f"Áreas temáticas: {areas_str}")
        parts.append(f"Intereses del estudiante: {body.user_interests}")
        query = "\n\n".join(parts)

        wu_top5 = await asyncio.to_thread(retriever.search_with_scores, query, 5)

        matched.append({
            "uam_code":   opt.code,
            "uam_name":   opt.name,
            "areas":      opt.areas_tematicas,
            "wu_matches": wu_top5,
        })

    return {"optatives_matched": matched}


@app.get("/search-wu")
def search_wu(q: str = "", limit: int = 10):
    """Búsqueda de asignaturas WU por nombre o código. Sin query devuelve las primeras 20."""
    chunks_path = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
    chunks_data: list = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not q.strip():
        subset = chunks_data[:20]
    else:
        q_lower = q.strip().lower()
        matches_exact = [c for c in chunks_data if c["code"] == q.strip()]
        matches_rest  = [
            c for c in chunks_data
            if c not in matches_exact
            and (q_lower in c["name"].lower() or q_lower in c["code"].lower())
        ]
        subset = (matches_exact + matches_rest)[:limit]

    return [
        {
            "code":     c["code"],
            "name":     c["name"],
            "credits":  c["metadata"].get("credits", ""),
            "type":     c["metadata"].get("type", ""),
            "schedule": c["metadata"].get("schedule", []),
        }
        for c in subset
    ]


@app.get("/health")
def health():
    return {"status": "ok"}
