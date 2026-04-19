"""
generator.py
------------
Generación de justificaciones de convalidación académica via Ollama Cloud (Gemma 4 E2B).

Función principal:
  generate_justification(query_course, top5_matches) → dict

Se hacen 5 llamadas secuenciales al LLM (una por candidata WU) y se agrega
el resultado en un único dict con 'matches' y 'summary'.

Dependencias: ollama
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path para imports relativos al proyecto
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
from ollama import Client

load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME      = "gpt-oss:120b"
TIMEOUT_SECONDS = 30
OUTPUT_PATH     = _PROJECT_ROOT / "data" / "processed" / "justification_result.json"
CHUNKS_PATH     = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
RETRIEVAL_TEST  = _PROJECT_ROOT / "data" / "processed" / "retrieval_test.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OLLAMA_API_KEY no está definida.\n"
            "Ejecútalo así:\n"
            "  export OLLAMA_API_KEY='tu_api_key'\n"
            "  python rag/generator.py"
        )
    log.info("Conectando a Ollama Cloud …")
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key},
    )


# ---------------------------------------------------------------------------
# Prompt builder — una llamada por match
# ---------------------------------------------------------------------------

def _build_prompt(query_course: dict, match: dict) -> str:
    """
    Construye el prompt para evaluar UN único match frente a la asignatura UAM.
    Trunca textos largos para no exceder el contexto del modelo.
    """

    def _trunc(text: str, n: int = 600) -> str:
        text = (text or "").strip()
        return text[:n] + "…" if len(text) > n else text

    qc     = query_course
    m      = match
    source = qc.get("source", "uam")

    # Bloque de la asignatura del estudiante: si es UAM solo mostramos contenidos
    if source == "uam":
        query_block = (
            f"Código: {qc.get('code', '?')}\n"
            f"Nombre: {qc.get('name', '?')}\n"
            f"ECTS: {qc.get('credits', '?')}\n"
            f"Contenidos:\n{_trunc(qc.get('contents', ''))}"
        )
    else:
        query_block = (
            f"Código: {qc.get('code', '?')}\n"
            f"Nombre: {qc.get('name', '?')}\n"
            f"ECTS: {qc.get('credits', '?')}\n"
            f"Contenidos:\n{_trunc(qc.get('contents', ''))}\n"
            f"Objetivos de aprendizaje:\n{_trunc(qc.get('learning_outcomes', ''))}"
        )

    # Los matches WU siempre incluyen contents + learning_outcomes
    wu_block = (
        f"Código: {m.get('code', '?')}\n"
        f"Nombre: {m.get('name', '?')}\n"
        f"ECTS: {m.get('credits', '?')} | Tipo: {m.get('type', '?')}\n"
        f"Contenidos:\n{_trunc(m.get('contents', ''), 500)}\n"
        f"Objetivos de aprendizaje:\n{_trunc(m.get('learning_outcomes', ''), 500)}"
    )

    return f"""Eres un experto en convalidación académica universitaria.

ASIGNATURA DEL ESTUDIANTE (Universidad Autónoma de Madrid):
{query_block}

ASIGNATURA WU VIENNA A EVALUAR:
{wu_block}

TAREA:
Analiza si la asignatura de WU Vienna es equivalente a la de la UAM para convalidación.

Responde EXCLUSIVAMENTE con un objeto JSON válido (sin texto antes ni después, sin bloques markdown) con esta estructura exacta:
{{
  "code": "{m.get('code', '?')}",
  "name": "{m.get('name', '?')}",
  "match_percentage": <número 0-100>,
  "overlapping_topics": ["<tema coincidente 1>", "<tema coincidente 2>"],
  "missing_in_wu": ["<tema que tiene UAM pero no WU>"],
  "extra_in_wu": ["<tema que tiene WU pero no UAM>"],
  "recommendation": "<SÍ convalidar|NO convalidar|REVISAR CON JEFE DE ESTUDIOS> - <razón breve en una frase>"
}}"""


# ---------------------------------------------------------------------------
# Llamada individual al LLM
# ---------------------------------------------------------------------------

def _call_one(client: Client, query_course: dict, match: dict) -> dict:
    """
    Llama al LLM para evaluar un único match.
    Devuelve el dict parseado o un dict con 'error' y 'raw_response'.
    """
    prompt = _build_prompt(query_course, match)
    code   = match.get("code", "?")
    log.info("  Evaluando match [%s] %s …", code, match.get("name", "?"))

    t0 = time.time()
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        elapsed = round(time.time() - t0, 2)
        raw = response["message"]["content"]
        log.info("  → Respuesta en %.1fs (%d chars)", elapsed, len(raw))

        parsed = _parse_json(raw)
        parsed["_elapsed"] = elapsed
        return parsed

    except Exception as exc:
        elapsed = round(time.time() - t0, 2)
        log.error("  → Error en [%s] (%.1fs): %s", code, elapsed, exc)
        return {
            "code":  code,
            "name":  match.get("name", "?"),
            "error": str(exc),
            "_elapsed": elapsed,
        }


# ---------------------------------------------------------------------------
# Parser JSON
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict:
    """
    Extrae JSON del texto del LLM.
    Intenta parseo directo → strip de fences markdown → bloque {…} más grande.
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw, flags=re.MULTILINE)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    log.warning("No se pudo parsear JSON — guardando raw_response.")
    return {"raw_response": raw, "parse_error": True}


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def generate_justification(
    query_course: dict,
    top5_matches: list[dict],
) -> dict:
    """
    5 llamadas secuenciales al LLM (una por candidata WU).

    Args:
        query_course: {code, name, credits, contents, learning_outcomes}
        top5_matches: lista de hasta 5 dicts {code, name, credits, type,
                      contents, learning_outcomes}

    Returns:
        {matches: [...], summary: "...", _meta: {...}}
    """
    log.info("Generando justificación para [%s] %s (%d candidatas) …",
             query_course.get("code"), query_course.get("name"), len(top5_matches))

    client  = _get_client()
    matches = []
    t_total = time.time()

    for i, match in enumerate(top5_matches[:5], 1):
        log.info("[%d/5] Procesando [%s] %s", i, match.get("code"), match.get("name"))
        result = _call_one(client, query_course, match)
        matches.append(result)

    elapsed_total = round(time.time() - t_total, 2)

    # Resumen automático basado en los resultados
    good = [m for m in matches if isinstance(m.get("match_percentage"), (int, float))
            and m["match_percentage"] >= 70]
    if good:
        best = max(good, key=lambda m: m["match_percentage"])
        summary = (
            f"Se encontraron {len(good)} asignatura(s) con similitud ≥70%. "
            f"La mejor opción es [{best.get('code')}] {best.get('name')} "
            f"con {best.get('match_percentage')}% de match. "
            f"Revisa las recomendaciones individuales para más detalle."
        )
    else:
        summary = (
            "Ninguna asignatura WU alcanza el 70% de similitud. "
            "Se recomienda revisar con el jefe de estudios antes de solicitar la convalidación."
        )

    return {
        "matches": matches,
        "summary": summary,
        "_meta": {
            "query_code":      query_course.get("code"),
            "query_name":      query_course.get("name"),
            "model":           MODEL_NAME,
            "total_elapsed_s": elapsed_total,
            "n_matches":       len(matches),
        },
    }


# ---------------------------------------------------------------------------
# Helpers para extraer texto de chunks.json
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^\[[A-Z_]+\]\s+\S+\s+\|\s*")

def _extract_section_text(course_entry: dict, section: str) -> str:
    """Concatena los chunks de una sección específica de un curso en chunks.json."""
    parts = [
        _PREFIX_RE.sub("", c["text"]).strip()
        for c in course_entry.get("chunks", [])
        if c.get("section") == section
    ]
    return "\n\n".join(parts)


def _course_entry_by_code(data: list[dict], code: str) -> dict | None:
    for entry in data:
        if entry.get("code") == code:
            return entry
    return None


# ---------------------------------------------------------------------------
# Print formateado
# ---------------------------------------------------------------------------

def _print_results(result: dict, query_name: str) -> None:
    matches  = result.get("matches", [])
    summary  = result.get("summary", "")
    meta     = result.get("_meta", {})

    print(f"\n{'='*68}")
    print(f"  JUSTIFICACIÓN DE CONVALIDACIÓN")
    print(f"  Query: {query_name}")
    print(f"  Modelo: {meta.get('model', MODEL_NAME)}  |  "
          f"Tiempo: {meta.get('total_elapsed_s', '?')}s")
    print(f"  {'─'*64}")

    if not matches:
        print("  Sin resultados (error o respuesta vacía).")
        if result.get("raw_response"):
            print(f"\n  Respuesta raw:\n  {result['raw_response'][:400]}")
        print(f"{'='*68}")
        return

    for m in matches:
        pct  = m.get("match_percentage", "?")
        rec  = m.get("recommendation", "")
        over = ", ".join(m.get("overlapping_topics", [])[:4])
        miss = ", ".join(m.get("missing_in_wu", [])[:3])
        xtra = ", ".join(m.get("extra_in_wu", [])[:3])

        # Colorear recomendación con símbolo ASCII
        rec_icon = "✓" if "SÍ" in rec else ("✗" if "NO" in rec else "~")

        print(f"\n  [{m.get('code','?')}] {m.get('name','?')}")
        print(f"   Match: {pct}%   {rec_icon} {rec}")
        if over:  print(f"   ↔ Coincide:  {over}")
        if miss:  print(f"   ← Falta WU:  {miss}")
        if xtra:  print(f"   → Extra WU:  {xtra}")

    if summary:
        print(f"\n  {'─'*64}")
        print(f"  RESUMEN: {summary}")

    print(f"\n{'='*68}")


# ---------------------------------------------------------------------------
# Script de prueba
# ---------------------------------------------------------------------------

def _run_test() -> None:
    """
    Query:   data/my_courses/16789.json  (source: uam)
    Matches: data/processed/retrieval_test.json  (top 5 más recientes)
    """
    my_course_path = _PROJECT_ROOT / "data" / "my_courses" / "16789.json"
    if not my_course_path.exists():
        log.error("No se encontró %s — ejecuta parse_my_syllabus.py primero.", my_course_path)
        return
    if not RETRIEVAL_TEST.exists():
        log.error("No se encontró %s — ejecuta retrieval.py primero.", RETRIEVAL_TEST)
        return
    if not CHUNKS_PATH.exists():
        log.error("No se encontró %s — ejecuta chunker.py primero.", CHUNKS_PATH)
        return

    # --- Query course desde my_courses/16789.json ---
    query_course = json.loads(my_course_path.read_text(encoding="utf-8"))
    query_code   = query_course.get("code", "16789")
    query_name   = query_course.get("name", "")
    log.info("Query: [%s] %s  (source: %s)", query_code, query_name, query_course.get("source"))

    # --- Top 5 desde el retrieval más reciente ---
    retrieval_data = json.loads(RETRIEVAL_TEST.read_text(encoding="utf-8"))
    top5_raw = retrieval_data.get("top_matches", [])
    log.info("Cargados %d matches desde %s", len(top5_raw), RETRIEVAL_TEST)

    # Enriquecer matches con contents + learning_outcomes desde chunks.json
    chunks_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    top5_matches: list[dict] = []
    for match in top5_raw[:5]:
        code  = match.get("code", "")
        entry = _course_entry_by_code(chunks_data, code)
        top5_matches.append({
            "code":              code,
            "name":              match.get("name", ""),
            "credits":           match.get("credits", ""),
            "type":              match.get("type", ""),
            "semantic_score":    match.get("semantic_score", 0),
            "bm25_score":        match.get("bm25_score", 0),
            "rrf_score":         match.get("rrf_score", 0),
            "contents":          _extract_section_text(entry, "contents") if entry else "",
            "learning_outcomes": _extract_section_text(entry, "learning_outcomes") if entry else "",
        })

    log.info("Generando justificación para [%s] %s con %d candidatas …",
             query_code, query_name, len(top5_matches))

    # --- Llamar al generador ---
    result = generate_justification(query_course, top5_matches)

    # --- Mostrar ---
    _print_results(result, f"[{query_code}] {query_name}")

    # --- Guardar ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Resultado guardado en %s", OUTPUT_PATH)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_test()
