"""
api.py
------
FastAPI backend para WU Matcher.

Endpoint:
  POST /upload-syllabus
    - Recibe un PDF de guía docente UAM
    - Ejecuta el pipeline completo: parse → retrieval → LLM justification
    - Retorna JSON con top5_matches y justificaciones

Ejecutar:
  uvicorn app.api:app --reload --port 8000

Dependencias: fastapi, uvicorn, python-multipart, pdfplumber
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from scraper.parse_my_syllabus import parse_pdf
from pipeline.retrieval import HybridRetriever
from rag.generator import generate_justification, _extract_section_text, _course_entry_by_code
import json

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="WU Matcher API",
    description="Encuentra equivalencias entre asignaturas UAM y WU Vienna",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Inicialización lazy del retriever (se carga una sola vez al primer request)
# ---------------------------------------------------------------------------

_retriever: HybridRetriever | None = None

def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/upload-syllabus")
async def upload_syllabus(file: UploadFile):
    """
    Pipeline completo:
      1. Guarda el PDF en un fichero temporal
      2. parse_my_syllabus.parse_pdf() → extrae code, name, contents
      3. retrieval.process_query_course() → top 5 matches semánticos
      4. generator.generate_justification() → justificaciones LLM
      5. Limpia el temporal y retorna JSON
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan ficheros PDF.")

    t_start = time.time()

    # --- 1. Guardar PDF temporal ---
    pdf_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = Path(tmp.name)

    try:
        # --- 2. Parsear PDF ---
        query_course = parse_pdf(tmp_path)

        if query_course.get("parse_error"):
            raise HTTPException(
                status_code=422,
                detail="No se pudo extraer información del PDF. ¿Es un plan docente UAM?",
            )

        # Asegurar source uam
        query_course.setdefault("source", "uam")

        # --- 3. Retrieval ---
        retriever = get_retriever()
        top5_raw  = retriever.process_query_course(query_course, top_courses=5)

        # --- 4. Enriquecer matches con contents + learning_outcomes desde chunks ---
        chunks_path = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
        chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

        top5_matches = []
        for match in top5_raw:
            code  = match.get("code", "")
            entry = _course_entry_by_code(chunks_data, code)
            top5_matches.append({
                **match,
                "contents":          _extract_section_text(entry, "contents") if entry else "",
                "learning_outcomes": _extract_section_text(entry, "learning_outcomes") if entry else "",
            })

        # --- 5. Generar justificaciones LLM ---
        justification = generate_justification(query_course, top5_matches)

        # Fusionar scores de retrieval con justificaciones del LLM
        llm_matches: dict[str, dict] = {
            m.get("code", ""): m
            for m in justification.get("matches", [])
        }

        final_matches = []
        for r in top5_raw:
            code = r.get("code", "")
            llm  = llm_matches.get(code, {})
            final_matches.append({
                "rank":               r["rank"],
                "code":               code,
                "name":               r.get("name", ""),
                "credits":            r.get("credits", ""),
                "type":               r.get("type", ""),
                "match_percentage":   llm.get("match_percentage", None),
                "recommendation":     llm.get("recommendation", ""),
                "overlapping_topics": llm.get("overlapping_topics", []),
                "missing_in_wu":      llm.get("missing_in_wu", []),
                "extra_in_wu":        llm.get("extra_in_wu", []),
                "error":              llm.get("error"),
            })

        processing_time = round(time.time() - t_start, 2)

        return {
            "status":        "success",
            "query_course":  {
                "code":    query_course.get("code", ""),
                "name":    query_course.get("name", ""),
                "credits": query_course.get("credits", ""),
            },
            "top5_matches":    final_matches,
            "summary":         justification.get("summary", ""),
            "processing_time": processing_time,
        }

    finally:
        # Limpiar siempre el fichero temporal
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}
