"""
classify_wu_courses.py
----------------------
Clasifica todos los cursos WU de data/raw/syllabi.json con el prompt
multi-label selectivo y guarda el resultado en
data/processed/wu_courses_classified.json.

Uso:
  python scripts/classify_wu_courses.py              # todos los cursos
  python scripts/classify_wu_courses.py --limit 10   # solo los primeros N
  python scripts/classify_wu_courses.py --code 1110  # solo un código
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
# Categorías
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

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

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
    return valid[:4]  # hard cap

def course_text(course: dict) -> str:
    sections = course.get("sections", {})
    parts = [
        sections[k]
        for k in ("contents", "learning_outcomes")
        if sections.get(k)
    ]
    return "\n\n".join(parts)[:1200]  # cap to avoid very long prompts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo los primeros N cursos")
    parser.add_argument("--code",  type=str, default=None, help="Procesar solo este código")
    parser.add_argument("--delay", type=float, default=0.3, help="Segundos entre llamadas al LLM")
    args = parser.parse_args()

    syllabi_path = _ROOT / "data" / "raw" / "syllabi.json"
    out_path     = _ROOT / "data" / "processed" / "wu_courses_classified.json"

    if not syllabi_path.exists():
        print(f"[ERROR] No se encontró {syllabi_path}", file=sys.stderr)
        sys.exit(1)

    data: list[dict] = json.loads(syllabi_path.read_text(encoding="utf-8"))

    # Filtros
    if args.code:
        data = [c for c in data if c["code"] == args.code]
        if not data:
            print(f"[ERROR] Código '{args.code}' no encontrado", file=sys.stderr)
            sys.exit(1)
    if args.limit:
        data = data[:args.limit]

    print(f"Clasificando {len(data)} cursos con modelo {MODEL_NAME}…\n")

    client  = _get_client()
    results = []
    errors  = []

    for i, course in enumerate(data, 1):
        code = course["code"]
        name = course["name"]
        text = course_text(course)

        if not text.strip():
            print(f"  [{i:>3}/{len(data)}] [{code}] {name[:55]} — SKIP (sin texto)")
            results.append({"code": code, "name": name, "areas": [], "error": "no_text"})
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
            results.append({"code": code, "name": name, "areas": areas})
        except Exception as exc:
            print(f"  [{i:>3}/{len(data)}] [{code}] {name[:55]} — ERROR: {exc}")
            errors.append({"code": code, "name": name, "error": str(exc)})
            results.append({"code": code, "name": name, "areas": [], "error": str(exc)})

        if args.delay and i < len(data):
            time.sleep(args.delay)

    # ── Stats ────────────────────────────────────────────────────────────────
    classified = [r for r in results if r["areas"]]
    avg_areas  = sum(len(r["areas"]) for r in classified) / max(len(classified), 1)

    print(f"\n{'─'*60}")
    print(f"Total cursos   : {len(data)}")
    print(f"Clasificados   : {len(classified)}")
    print(f"Errores/vacíos : {len(results) - len(classified)}")
    print(f"Avg áreas/curso: {avg_areas:.2f}")

    # Frecuencia de cada área
    from collections import Counter
    freq = Counter(a for r in classified for a in r["areas"])
    print("\nFrecuencia de áreas:")
    for area, count in freq.most_common():
        bar = "█" * (count * 20 // max(freq.values(), default=1))
        print(f"  {area:<20} {count:>3}  {bar}")

    # ── Guardar ──────────────────────────────────────────────────────────────
    output = {
        "model":   MODEL_NAME,
        "total":   len(data),
        "areas":   AREAS,
        "courses": results,
        "errors":  errors,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado en: {out_path}")


if __name__ == "__main__":
    main()
