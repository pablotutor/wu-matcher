"""
debug_classify.py
-----------------
Debug de clasificación multi-label para el curso E&I Zone: AI Bootcamp for Entrepreneurs.

Uso:
  python scripts/debug_classify.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from rag.generator import _get_client, MODEL_NAME

# ---------------------------------------------------------------------------
# Categorías extendidas
# ---------------------------------------------------------------------------

AREAS = [
    "MARKETING", "FINANZAS", "DATA_SCIENCE", "GESTIÓN", "ESTRATEGIA",
    "DERECHO", "TECNOLOGÍA", "SOSTENIBILIDAD", "RECURSOS_HUMANOS",
    "ENTREPRENEURSHIP", "IA", "PRODUCT",
]
AREAS_STR = ", ".join(AREAS)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXAMPLES = """\
Ejemplos:
- "AI Bootcamp for Entrepreneurs" → IA, ENTREPRENEURSHIP, PRODUCT
- "Financial Analysis with Machine Learning" → FINANZAS, IA, DATA_SCIENCE
- "Sustainable Business Strategy" → SOSTENIBILIDAD, ESTRATEGIA"""

def prompt_selective(text: str) -> str:
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

def prompt_flexible(text: str) -> str:
    return (
        f"¿A cuántas de estas áreas pertenece esta asignatura? Selecciona TODAS las que apliquen:\n"
        f"{AREAS_STR}.\n\n"
        f"{EXAMPLES}\n\n"
        f"Devuelve únicamente los nombres aplicables separados por comas, sin texto adicional.\n\n"
        f"Contenido:\n{text}"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 70

def parse_areas(raw: str) -> list[str]:
    tokens = [t.strip().upper() for t in raw.replace("\n", ",").split(",")]
    return [t for t in tokens if t in AREAS]

def call_llm(client, prompt: str, label: str) -> tuple[str, list[str]]:
    print(f"\n{'─'*4} {label} {'─'*4}")
    print("PROMPT (últimas 3 líneas):")
    for line in prompt.strip().splitlines()[-3:]:
        print(f"  {line}")
    print()

    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.message.content.strip()
    parsed = parse_areas(raw)

    print(f"RAW OUTPUT  : {raw!r}")
    print(f"PARSED AREAS: {parsed}")
    print(f"COUNT       : {len(parsed)}")
    return raw, parsed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    syllabi_path = _ROOT / "data" / "raw" / "syllabi.json"
    if not syllabi_path.exists():
        print(f"[ERROR] No se encontró {syllabi_path}", file=sys.stderr)
        sys.exit(1)

    data: list[dict] = json.loads(syllabi_path.read_text(encoding="utf-8"))

    # Target: E&I Zone: AI Bootcamp for Entrepreneurs (code 1110)
    target_name = "E&I Zone"
    target_sub  = "Artificial Intelligence"
    course = next(
        (c for c in data
         if target_name in c.get("name", "") and "AI" in c.get("name", "")),
        None,
    )
    if course is None:
        # fallback: first E&I Zone course
        course = next((c for c in data if target_name in c.get("name", "")), None)
    if course is None:
        print(f"[ERROR] No se encontró ningún curso '{target_name}'", file=sys.stderr)
        print("Cursos disponibles:")
        for c in data[:20]:
            print(f"  [{c['code']}] {c['name']}")
        sys.exit(1)

    # ── Print full course text ───────────────────────────────────────────────
    print(DIVIDER)
    print(f"CURSO  : [{course['code']}] {course['name']}")
    print(f"CRÉDITS: {course.get('credits', '?')} · {course.get('type', '?')}")
    print(DIVIDER)

    sections = course.get("sections", {})
    full_text_parts: list[str] = []
    for section, content in sections.items():
        if section in ("contents", "learning_outcomes"):
            print(f"\n{'━'*4} {section.upper()} {'━'*4}")
            print(content)
            full_text_parts.append(content)

    full_text = "\n\n".join(full_text_parts)
    print(f"\n[INFO] Texto enviado al LLM: {len(full_text)} caracteres")

    # ── LLM calls ───────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("CLASIFICACIÓN MULTI-LABEL")
    print(DIVIDER)

    client = _get_client()

    _, areas_selective = call_llm(client, prompt_selective(full_text), "PROMPT SELECTIVO (nuevo)")

    if len(areas_selective) <= 1:
        print("\n[WARN] Devolvió ≤1 categoría. Probando prompt flexible como fallback...")
        _, areas_fallback = call_llm(client, prompt_flexible(full_text), "PROMPT FLEXIBLE (fallback)")
    else:
        areas_fallback = areas_selective
        print("\n[OK] Prompt selectivo devolvió múltiples categorías.")

    # ── Resultado final ──────────────────────────────────────────────────────
    best = areas_fallback if len(areas_fallback) >= len(areas_selective) else areas_selective
    expected = {"IA", "ENTREPRENEURSHIP", "PRODUCT"}
    match    = expected.issubset(set(best))

    print(f"\n{DIVIDER}")
    print(f"RESULTADO FINAL : {best}")
    print(f"ESPERADO        : {sorted(expected)}")
    print(f"VALIDACIÓN      : {'✓ OK' if match else '✗ FALTA ' + str(expected - set(best))}")
    if len(best) > 4:
        print(f"[WARN] Más de 4 categorías ({len(best)}): el modelo ignoró el límite")
    print(DIVIDER)

    # ── Guardar resultado ────────────────────────────────────────────────────
    out_path = _ROOT / "data" / "processed" / "debug_classify_result.json"
    result = {
        "course_code":      course["code"],
        "course_name":      course["name"],
        "areas_selective":  areas_selective,
        "areas_fallback":   areas_fallback,
        "best":             best,
        "expected":         sorted(expected),
        "validation_ok":    match,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResultado guardado en: {out_path}")


if __name__ == "__main__":
    main()
