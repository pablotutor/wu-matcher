"""
parse_my_syllabus.py
--------------------
Extrae campos estructurados de un plan docente UAM (PDF) en DOS fases:

  Fase A — Regex directa sobre el texto del PDF:
    • contents        → sección 1.13 (entre "1.13" y "1.14" / "Referencias")
    • learning_outcomes → sección 1.12 (entre "1.12" y "1.13")

  Fase B — LLM (Ollama Cloud) solo para metadata ligera:
    • code, name, credits, language   (prompt corto, ~2 000 chars)

La extracción estructural (contenidos largos) nunca pasa por el LLM,
lo que evita truncados y alucinaciones en campos largos.

Uso:
  python scraper/parse_my_syllabus.py ruta/al/plan_docente.pdf
  python scraper/parse_my_syllabus.py        # todos los PDFs en data/my_courses/

Dependencias: pdfplumber, ollama, python-dotenv
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv
from ollama import Client

# ---------------------------------------------------------------------------
# Paths y config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

MY_COURSES_DIR = _PROJECT_ROOT / "data" / "my_courses"
MODEL_NAME     = "gpt-oss:120b"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cliente Ollama
# ---------------------------------------------------------------------------

def _get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OLLAMA_API_KEY no está definida.\n"
            "  export OLLAMA_API_KEY='tu_api_key'"
        )
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key},
    )


# ---------------------------------------------------------------------------
# Extracción de texto del PDF
# ---------------------------------------------------------------------------

_FOOTER_PATTERNS = [
    re.compile(r"^Código Seguro de Verificación[^\n]*$", re.MULTILINE),
    re.compile(r"^Firmado por:[^\n]*$", re.MULTILINE),
    re.compile(r"^\d+/\d+$", re.MULTILINE),
    re.compile(r"^Url de Verificación:[^\n]*$", re.MULTILINE),
]


def _strip_page_footers(text: str) -> str:
    """Elimina las líneas de pie de página estándar UAM."""
    for pat in _FOOTER_PATTERNS:
        text = pat.sub("", text)
    # Colapsar líneas en blanco consecutivas generadas al eliminar footers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_text(pdf_path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        log.info("PDF abierto: %d páginas.", len(pdf.pages))
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=3)
            if text:
                pages.append(_strip_page_footers(text))
    full_text = "\n\n".join(p for p in pages if p)
    log.info("Texto extraído: %d caracteres.", len(full_text))
    return full_text


# ---------------------------------------------------------------------------
# FASE A — Extracción regex de secciones largas
# ---------------------------------------------------------------------------

def _regex_extract_contents(text: str) -> str:
    """
    Extrae sección 1.13 (Contenidos) con regex.
    Captura todo entre el header "1.13" y la siguiente sección numerada
    (1.14, 1.15, …), "Referencias", "Bibliografía" o fin de texto.
    """
    # Stop al llegar a 1.14+, a una sección principal nueva (2. / 3. …) o a referencias.
    # NO paramos en 1.X. con X<14 porque esos son subtemas dentro del propio temario.
    _STOP = r"(?=\n1\.1[4-9]\.?[\s\n]|\n1\.[2-9]\d\.?[\s\n]|\n\d{1,2}\.\s+[A-ZÁÉÍÓÚ]|\nReferencias|\nBibliograf|\Z)"
    patterns = [
        # Formato estándar UAM: "1.13 Contenidos del programa" o "1.13. Contenidos"
        r"1\.13\.?\s*[Cc]ontenidos[^\n]*\n(.*?)" + _STOP,
        # Fallback: "Contenidos del programa" sin número de sección
        r"Contenidos del programa[^\n]*\n(.*?)" + _STOP,
        # Fallback: "Temario" o "Programa detallado"
        r"(?:Temario|Programa detallado)[^\n]*\n(.*?)" + _STOP,
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result = m.group(1).strip()
            if len(result) > 50:   # descartar capturas vacías o triviales
                log.info("contents extraído por regex: %d chars.", len(result))
                return result
    log.warning("No se encontró sección 1.13 con regex.")
    return ""


def _regex_extract_outcomes(text: str) -> str:
    """
    Extrae sección 1.12 (Competencias / Resultados de aprendizaje) con regex.
    Captura todo entre "1.12" y "1.13".
    """
    patterns = [
        r"1\.12\.?\s*(?:Competencias|Resultados)[^\n]*\n(.*?)(?=\n1\.13)",
        r"1\.12\.?\s*[^\n]*\n(.*?)(?=\n1\.13)",
        r"(?:Competencias|Resultados de aprendizaje)[^\n]*\n(.*?)(?=\n1\.13|\n1\.1[4-9]|\Z)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            result = m.group(1).strip()
            if len(result) > 30:
                log.info("learning_outcomes extraído por regex: %d chars.", len(result))
                return result
    log.warning("No se encontró sección 1.12 con regex.")
    return ""


# ---------------------------------------------------------------------------
# FASE B — LLM solo para metadata corta
# ---------------------------------------------------------------------------

def _build_metadata_prompt(text: str) -> str:
    """
    Prompt corto: solo pide code, name, credits, language.
    Usa solo las primeras ~2 000 chars (donde suele estar la portada/cabecera).
    """
    header = text[:2_000]
    return f"""Eres un extractor de información académica española.

Del siguiente fragmento de un plan docente de la UAM, extrae SOLO estos 4 campos en JSON:
{{
  "code": "código numérico de la asignatura (ej: 16767)",
  "name": "nombre completo de la asignatura en mayúsculas",
  "credits": "créditos ECTS como string (ej: '6')",
  "language": "idioma de impartición (ej: 'Español', 'Inglés')"
}}

Devuelve SOLO el JSON, sin texto adicional ni bloques markdown.

Fragmento del plan docente:
{header}"""


def _call_llm_metadata(client: Client, text: str) -> dict:
    prompt = _build_metadata_prompt(text)
    log.info("Llamando al LLM para metadata (%d chars de prompt) …", len(prompt))
    response = client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    raw = response["message"]["content"]
    log.info("Respuesta LLM: %d chars.", len(raw))
    return _parse_json_response(raw)


# ---------------------------------------------------------------------------
# Parser JSON
# ---------------------------------------------------------------------------

def _parse_json_response(raw: str) -> dict:
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

def parse_pdf(pdf_path: str | Path) -> dict:
    """
    Extrae campos estructurados de un plan docente UAM en PDF.

    Fase A (regex): contents, learning_outcomes
    Fase B (LLM):   code, name, credits, language
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    log.info("── Procesando: %s ──", pdf_path.name)

    # ── Extraer texto del PDF ──────────────────────────────────────────────
    text = _extract_pdf_text(pdf_path)
    if not text.strip():
        raise ValueError(f"No se pudo extraer texto de: {pdf_path}")

    # ── FASE A: regex ──────────────────────────────────────────────────────
    log.info("Fase A — extracción por regex …")
    contents         = _regex_extract_contents(text)
    learning_outcomes = _regex_extract_outcomes(text)

    # Debug: primeros 200 chars de lo extraído
    print(f"\n  [DEBUG] contents  ({len(contents)} chars): "
          f"\"{contents[:200].replace(chr(10), ' ')}\"")
    print(f"  [DEBUG] outcomes  ({len(learning_outcomes)} chars): "
          f"\"{learning_outcomes[:200].replace(chr(10), ' ')}\"\n")

    # ── FASE B: LLM para metadata ──────────────────────────────────────────
    log.info("Fase B — extracción de metadata via LLM …")
    client   = _get_client()
    metadata = _call_llm_metadata(client, text)

    if metadata.get("parse_error"):
        log.error("LLM no devolvió JSON válido para metadata.")
        metadata = {"code": pdf_path.stem, "name": "", "credits": "", "language": ""}

    # ── Combinar ───────────────────────────────────────────────────────────
    data = {
        "code":     metadata.get("code",     pdf_path.stem).strip(),
        "name":     metadata.get("name",     "").strip(),
        "credits":  metadata.get("credits",  "").strip(),
        "language": metadata.get("language", "").strip(),
        "contents": contents,
        "source":   "uam",
    }

    # ── Guardar ────────────────────────────────────────────────────────────
    MY_COURSES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MY_COURSES_DIR / f"{data['code']}.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Guardado en: %s", out_path)

    _print_summary(data)
    return data


# ---------------------------------------------------------------------------
# Print resumen
# ---------------------------------------------------------------------------

def _print_summary(data: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  CAMPOS EXTRAÍDOS")
    print(f"  {'─'*56}")
    fields = [
        ("code",              "Código"),
        ("name",              "Nombre"),
        ("credits",           "ECTS"),
        ("language",          "Idioma"),
        ("contents",          "Contenidos"),
        ("learning_outcomes", "Resultados aprendizaje"),
    ]
    for key, label in fields:
        value = data.get(key, "")
        length = len(value)
        preview = value[:80].replace("\n", " ") + ("…" if length > 80 else "")
        if length > 80:
            print(f"  {label:<26} ({length} chars)")
            print(f"    └─ \"{preview}\"")
        else:
            print(f"  {label:<26} \"{preview}\"")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        pdf_paths = [Path(sys.argv[1])]
    else:
        MY_COURSES_DIR.mkdir(parents=True, exist_ok=True)
        pdf_paths = sorted(MY_COURSES_DIR.glob("*.pdf"))
        if not pdf_paths:
            print(f"No se encontraron PDFs en {MY_COURSES_DIR}")
            print("Uso: python scraper/parse_my_syllabus.py ruta/al/plan_docente.pdf")
            return

    for pdf_path in pdf_paths:
        print(f"\nProcesando: {pdf_path}")
        try:
            result = parse_pdf(pdf_path)
            if result.get("parse_error"):
                print(f"  ERROR: {result.get('raw_response', '')[:500]}")
        except Exception as exc:
            log.error("Error procesando %s: %s", pdf_path.name, exc)


if __name__ == "__main__":
    main()
