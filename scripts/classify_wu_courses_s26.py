"""
classify_wu_courses_s26.py
--------------------------
Clasifica los cursos de Summer 2026 (data/raw/syllabi_s26.json) y fusiona
los resultados en data/processed/wu_courses_classified.json sin borrar las
clasificaciones de Winter 2025.

Si un código ya existe en el JSON, lo actualiza; si es nuevo, lo añade.

Uso:
  python scripts/classify_wu_courses_s26.py            # todos los cursos s26
  python scripts/classify_wu_courses_s26.py --limit 10
  python scripts/classify_wu_courses_s26.py --code 1110
  python scripts/classify_wu_courses_s26.py --delay 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from rag.generator import _get_client, MODEL_NAME

# ---------------------------------------------------------------------------
# Categorías (mismas que classify_wu_courses.py)
# ---------------------------------------------------------------------------

AREAS = [
    "MARKETING", "FINANZAS", "DATA_SCIENCE", "GESTIÓN", "ESTRATEGIA",
    "DERECHO", "TECNOLOGÍA", "SOSTENIBILIDAD", "RECURSOS_HUMANOS",
    "ENTREPRENEURSHIP", "IA", "PRODUCT",
]
AREAS_STR = ", ".join(AREAS)
AREAS_SET  = set(AREAS)

EXAMPLES = """\
Ejemplos:
- "AI Bootcamp for Entrepreneurs" → IA, ENTREPRENEURSHIP, PRODUCT
- "Financial Analysis with Machine Learning" → FINANZAS, IA, DATA_SCIENCE
- "Sustainable Business Strategy" → SOSTENIBILIDAD, ESTRATEGIA"""


def build_prompt(text: str) -> str:
    return (
        f"Dado el contenido de una asignatura, clasifícala SOLO en las categorías "
        f"PRINCIPALES y DIRECTAS (máximo 3-4).\n\n"
        f"Categorías disponibles:\n{AREAS_STR}.\n\n"
        f"REGLAS:\n"
        f"- Solo incluye categorías que sean EXPLÍCITAMENTE mencionadas o muy claramente implícitas en el contenido\n"
        f"- Si es sobre \"IA aplicada a X\", incluye AMBAS: IA + la aplicación (X)\n"
        f"- No incluyas categorías tangenciales o remotamente relacionadas\n"
        f"- Si duda entre dos, elige la más específica\n\n"
        f"{EXAMPLES}\n\n"
        f"Devuelve SOLO los nombres separados por comas. Máximo 4 categorías.\n\n"
        f"Contenido de la asignatura:\n{text}"
    )


def parse_areas(raw: str) -> list[str]:
    tokens = [t.strip().upper() for t in raw.replace("\n", ",").split(",")]
    valid  = [t for t in tokens if t in AREAS_SET]
    return valid[:4]


def course_text(course: dict) -> str:
    sections = course.get("sections", {})
    parts = [
        sections[k]
        for k in ("contents", "learning_outcomes")
        if sections.get(k)
    ]
    return "\n\n".join(parts)[:1200]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int,   default=None)
    parser.add_argument("--code",  type=str,   default=None)
    parser.add_argument("--delay", type=float, default=0.3)
    args = parser.parse_args()

    syllabi_path = _ROOT / "data" / "raw" / "syllabi_s26.json"
    out_path     = _ROOT / "data" / "processed" / "wu_courses_classified.json"

    if not syllabi_path.exists():
        print(f"[ERROR] No se encontró {syllabi_path} — ejecuta phase2_syllabi_s26.py primero.",
              file=sys.stderr)
        sys.exit(1)

    data: list[dict] = json.loads(syllabi_path.read_text(encoding="utf-8"))

    # Saltar cursos con error de scraping
    data = [c for c in data if c.get("sections", {}).get("status", "ok") == "ok"]

    if args.code:
        data = [c for c in data if c["code"] == args.code]
        if not data:
            print(f"[ERROR] Código '{args.code}' no encontrado", file=sys.stderr)
            sys.exit(1)
    if args.limit:
        data = data[:args.limit]

    print(f"Clasificando {len(data)} cursos de Summer 2026 con modelo {MODEL_NAME}…\n")

    client  = _get_client()
    new_results: list[dict] = []
    errors:      list[dict] = []

    for i, course in enumerate(data, 1):
        code = course["code"]
        name = course["name"]
        text = course_text(course)

        if not text.strip():
            print(f"  [{i:>3}/{len(data)}] [{code}] {name[:55]} — SKIP (sin texto)")
            new_results.append({"code": code, "name": name, "areas": [], "error": "no_text"})
            continue

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": build_prompt(text)}],
            )
            raw    = response.message.content.strip()
            areas  = parse_areas(raw)
            status = f"{areas}" if areas else f"[WARN sin áreas válidas] raw={raw!r}"
            print(f"  [{i:>3}/{len(data)}] [{code}] {name[:55]:<55} → {status}")
            new_results.append({"code": code, "name": name, "areas": areas})
        except Exception as exc:
            print(f"  [{i:>3}/{len(data)}] [{code}] {name[:55]} — ERROR: {exc}")
            errors.append({"code": code, "name": name, "error": str(exc)})
            new_results.append({"code": code, "name": name, "areas": [], "error": str(exc)})

        if args.delay and i < len(data):
            time.sleep(args.delay)

    # ── Fusionar con clasificaciones existentes ──────────────────────────────
    if out_path.exists():
        existing_data = json.loads(out_path.read_text(encoding="utf-8"))
        existing_courses = existing_data.get("courses", [])
    else:
        existing_data    = {"model": MODEL_NAME, "areas": AREAS, "courses": [], "errors": []}
        existing_courses = []

    existing_by_code = {c["code"]: i for i, c in enumerate(existing_courses)}
    added = updated = 0

    for course in new_results:
        code = course["code"]
        if code in existing_by_code:
            existing_courses[existing_by_code[code]] = course
            updated += 1
        else:
            existing_courses.append(course)
            added += 1

    all_errors = existing_data.get("errors", []) + errors

    output = {
        "model":   MODEL_NAME,
        "total":   len(existing_courses),
        "areas":   AREAS,
        "courses": existing_courses,
        "errors":  all_errors,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Stats ────────────────────────────────────────────────────────────────
    classified = [r for r in new_results if r.get("areas")]
    print(f"\n{'─'*60}")
    print(f"Cursos s26 procesados : {len(data)}")
    print(f"Clasificados          : {len(classified)}")
    print(f"Errores/vacíos        : {len(new_results) - len(classified)}")
    print(f"Añadidos al JSON      : {added}")
    print(f"Actualizados en JSON  : {updated}")
    print(f"Total en classified   : {len(existing_courses)}")
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
