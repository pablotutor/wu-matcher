"""
chunker.py
----------
Chunking estructural por secciones: divide cada syllabus en fragmentos
semánticamente coherentes según las reglas de cada tipo de sección.

Reglas de chunking:
  CONTENTS            → párrafos naturales (\n\n); si > 200 palabras, por oraciones; max 300 tokens
  LEARNING_OUTCOMES   → líneas (bullets/números); párrafos continuos → por oraciones
  ATTENDANCE          → un chunk; si múltiples reglas, divide por regla
  TEACHING_METHODS    → por método; si párrafo, por oraciones
  ASSESSMENT          → por criterio de evaluación; mantiene % con criterio
  SCHEDULE            → no se chunka, va a metadata

Dependencias: ninguna externa (stdlib json, pathlib, re)
"""

import json
import re
import logging
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYLLABI_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "syllabi.json"
OUTPUT_PATH  = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"

MAX_TOKENS_CONTENTS = 300
LONG_PARAGRAPH_WORDS = 200

SECTION_PREFIXES = {
    "contents":               "CONTENTS",
    "learning_outcomes":      "LEARNING_OUTCOMES",
    "attendance_requirements": "ATTENDANCE",
    "teaching_methods":       "TEACHING_METHODS",
    "assessment":             "ASSESSMENT",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de texto
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Elimina HTML residual, colapsa espacios y normaliza líneas."""
    # Eliminar tags HTML residuales
    text = re.sub(r"<[^>]+>", " ", text)
    # Colapsar múltiples espacios/tabs en uno
    text = re.sub(r"[ \t]+", " ", text)
    # Colapsar más de dos saltos de línea consecutivos
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    """Aproxima el número de tokens: palabras / 0.75."""
    return round(len(text.split()) / 0.75)


# Abreviaturas frecuentes para no cortar en su punto final
_ABBREVS = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e|Fig|No|Vol|pp|Ch|Sec|al|cf|approx|dept)\.",
    re.IGNORECASE,
)

def split_by_sentences(text: str) -> list[str]:
    """
    Divide texto en oraciones respetando . ! ? pero no abreviaturas comunes.
    Devuelve lista de oraciones no vacías.
    """
    # Proteger abreviaturas sustituyendo su punto por un placeholder
    protected = _ABBREVS.sub(lambda m: m.group(0).replace(".", "<<<DOT>>>"), text)
    # Dividir en . ! ? seguidos de espacio o fin de cadena
    raw = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [s.replace("<<<DOT>>>", ".").strip() for s in raw if s.strip()]
    return sentences


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    """
    Divide `text` en chunks de hasta `max_tokens`.
    Intenta no partir a mitad de oración: acumula oraciones hasta el límite.
    """
    sentences = split_by_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sent in sentences:
        t = count_tokens(sent)
        if current_tokens + t > max_tokens and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_tokens = t
        else:
            current.append(sent)
            current_tokens += t

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Chunkers por sección
# ---------------------------------------------------------------------------

def _chunk_contents(text: str, code: str) -> list[str]:
    """
    Divide por párrafos naturales (\n\n).
    Si un párrafo supera LONG_PARAGRAPH_WORDS palabras, lo subdivide por oraciones.
    Aplica límite de MAX_TOKENS_CONTENTS por chunk.
    """
    prefix = f"[CONTENTS] {code} |"
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    result: list[str] = []

    for para in paragraphs:
        words = len(para.split())
        if words > LONG_PARAGRAPH_WORDS or count_tokens(para) > MAX_TOKENS_CONTENTS:
            sub_chunks = _split_long_text(para, MAX_TOKENS_CONTENTS)
        else:
            sub_chunks = [para]

        for chunk in sub_chunks:
            result.append(f"{prefix} {chunk}")

    return result


def _chunk_learning_outcomes(text: str, code: str) -> list[str]:
    """
    Divide por líneas (bullets / números).
    Si hay párrafos continuos (sin saltos de línea entre ítems), divide por oraciones.
    """
    prefix = f"[LEARNING_OUTCOMES] {code} |"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Determinar si parece una lista (muchas líneas cortas) o párrafo continuo
    avg_words = sum(len(l.split()) for l in lines) / max(len(lines), 1)
    result: list[str] = []

    if avg_words < 25:
        # Lista: cada línea es un chunk
        for line in lines:
            result.append(f"{prefix} {line}")
    else:
        # Párrafo continuo: dividir por oraciones
        full_text = " ".join(lines)
        for sent in split_by_sentences(full_text):
            result.append(f"{prefix} {sent}")

    return result


def _chunk_attendance(text: str, code: str) -> list[str]:
    """
    Generalmente un único chunk.
    Si hay múltiples líneas/reglas, divide por regla (línea).
    """
    prefix = f"[ATTENDANCE] {code} |"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if len(lines) <= 1:
        return [f"{prefix} {text.strip()}"] if text.strip() else []

    # Identificar si las líneas parecen reglas independientes (cortas, quizás numeradas)
    result: list[str] = []
    current_rule: list[str] = []

    for line in lines:
        # Nueva regla si empieza con número/bullet o la línea anterior era larga
        if re.match(r"^[\d\-•*]\s", line) or (current_rule and len(current_rule[-1].split()) > 10):
            if current_rule:
                result.append(f"{prefix} {' '.join(current_rule)}")
            current_rule = [line]
        else:
            current_rule.append(line)

    if current_rule:
        result.append(f"{prefix} {' '.join(current_rule)}")

    return result if result else [f"{prefix} {text.strip()}"]


def _chunk_teaching_methods(text: str, code: str) -> list[str]:
    """
    Divide por método mencionado.
    Si el texto es un párrafo continuo, divide por oraciones.
    """
    prefix = f"[TEACHING_METHODS] {code} |"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Keywords que suelen delimitar métodos de enseñanza
    method_keywords = re.compile(
        r"\b(lecturing|lecture|group work|discussion|case stud|seminar|"
        r"presentation|workshop|tutorial|exercise|problem.solving|"
        r"project|simulation|role.play|e-learning|online|flipped)\b",
        re.IGNORECASE,
    )

    # Si hay varias líneas cortas que mencionan métodos, cada línea = un chunk
    if len(lines) > 1 and all(len(l.split()) < 30 for l in lines):
        return [f"{prefix} {line}" for line in lines]

    # Si hay líneas con keywords al inicio, dividir por ellas
    full = " ".join(lines)
    sentences = split_by_sentences(full)
    result: list[str] = []
    current: list[str] = []

    for sent in sentences:
        if method_keywords.search(sent) and current:
            result.append(f"{prefix} {' '.join(current)}")
            current = [sent]
        else:
            current.append(sent)

    if current:
        result.append(f"{prefix} {' '.join(current)}")

    return result if result else [f"{prefix} {text.strip()}"]


def _chunk_assessment(text: str, code: str) -> list[str]:
    """
    Divide por criterio de evaluación (Exam 50%, Project 25%, etc.).
    Mantiene el % junto con su criterio.
    """
    prefix = f"[ASSESSMENT] {code} |"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Patrón de criterio: línea con porcentaje o línea que empieza con keyword de evaluación
    criterion_start = re.compile(
        r"(?:\d+\s*%|^\d+\.|\b(?:exam|test|quiz|project|assignment|participation|"
        r"presentation|paper|essay|homework|midterm|final|written|oral|group|"
        r"individual|continuous|attendance)\b)",
        re.IGNORECASE,
    )

    result: list[str] = []
    current: list[str] = []

    for line in lines:
        if criterion_start.search(line) and current:
            result.append(f"{prefix} {' '.join(current)}")
            current = [line]
        else:
            current.append(line)

    if current:
        result.append(f"{prefix} {' '.join(current)}")

    # Si no se detectaron criterios claros, intentar dividir por oraciones
    if len(result) == 1 and not criterion_start.search(text):
        full = " ".join(lines)
        sentences = split_by_sentences(full)
        if len(sentences) > 1:
            return [f"{prefix} {s}" for s in sentences]

    return result if result else [f"{prefix} {text.strip()}"]


# ---------------------------------------------------------------------------
# Dispatcher por sección
# ---------------------------------------------------------------------------

_CHUNKERS = {
    "contents":               _chunk_contents,
    "learning_outcomes":      _chunk_learning_outcomes,
    "attendance_requirements": _chunk_attendance,
    "teaching_methods":       _chunk_teaching_methods,
    "assessment":             _chunk_assessment,
}


# ---------------------------------------------------------------------------
# Procesador de un syllabus completo
# ---------------------------------------------------------------------------

def process_syllabus(syllabus: dict) -> dict:
    """
    Convierte un syllabus scrapeado en el formato de chunks procesados.
    Devuelve el dict con 'code', 'name', 'chunks' y 'metadata'.
    """
    code     = syllabus.get("code", "")
    name     = syllabus.get("name", "")
    sections = syllabus.get("sections", {})
    schedule = syllabus.get("schedule", [])
    credits  = syllabus.get("credits", "")
    ctype    = syllabus.get("type", "")

    all_chunks: list[dict] = []
    counters: dict[str, int] = defaultdict(int)

    for section_key, chunker_fn in _CHUNKERS.items():
        raw_text = sections.get(section_key, "")
        if not raw_text:
            continue

        cleaned = _clean(raw_text)
        if not cleaned:
            continue

        texts = chunker_fn(cleaned, code)

        for text in texts:
            text = text.strip()
            if not text:
                continue
            counters[section_key] += 1
            chunk_id = f"{code}_{section_key}_{counters[section_key]:03d}"
            all_chunks.append({
                "id":               chunk_id,
                "section":          section_key,
                "text":             text,
                "tokens_estimated": count_tokens(text),
            })

    return {
        "code":   code,
        "name":   name,
        "chunks": all_chunks,
        "metadata": {
            "schedule": schedule,
            "credits":  credits,
            "type":     ctype,
        },
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not SYLLABI_PATH.exists():
        log.error("No se encontró %s — ejecuta phase2_syllabi.py primero.", SYLLABI_PATH)
        return

    syllabi = json.loads(SYLLABI_PATH.read_text(encoding="utf-8"))
    log.info("Cargados %d syllabi desde %s", len(syllabi), SYLLABI_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed: list[dict] = []
    total_chunks = 0
    section_dist: dict[str, int] = defaultdict(int)
    total_tokens = 0

    for syl in syllabi:
        # Saltar cursos sin secciones o con error
        status = syl.get("sections", {}).get("status", "")
        if status in ("error", "404", "no_url"):
            continue

        doc = process_syllabus(syl)
        processed.append(doc)

        for chunk in doc["chunks"]:
            total_chunks += 1
            section_dist[chunk["section"]] += 1
            total_tokens += chunk["tokens_estimated"]

    OUTPUT_PATH.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    avg_tokens = round(total_tokens / total_chunks) if total_chunks else 0

    print(f"\n{'='*55}")
    print(f"  Syllabi procesados:       {len(processed)}")
    print(f"  Total chunks generados:   {total_chunks}")
    print(f"  Promedio tokens/chunk:    {avg_tokens}")
    print(f"\n  Distribución por sección:")
    section_order = [
        "contents", "learning_outcomes", "attendance_requirements",
        "teaching_methods", "assessment",
    ]
    for sec in section_order:
        n = section_dist.get(sec, 0)
        label = SECTION_PREFIXES.get(sec, sec)
        print(f"    {label:<22}  {n:>5} chunks")
    print(f"\n  Resultado en:             {OUTPUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
