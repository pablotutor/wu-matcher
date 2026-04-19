"""
phase2_syllabi.py
-----------------
Scraper de fase 2: extrae el contenido completo de cada syllabus de la WU Vienna
a partir de la lista de cursos generada por phase1_catalog.py.

Estructura HTML real confirmada (Bootstrap panels):

    <div class="panel panel-default card mb-3">
      <div class="panel-heading card-header">
        <span class="bold"><a name="lvbeschreibungN"></a>Section Title</span>
      </div>
      <div class="panel-body card-body">
        <p><span class="text">Contenido...</span></p>
      </div>
    </div>

Mapa de anchors:
    lvbeschreibung1 → contents
    lvbeschreibung2 → learning_outcomes
    lvbeschreibung3 → attendance_requirements
    lvbeschreibung4 → teaching_methods
    lvbeschreibung5 → assessment

Tabla de horarios: <table summary="Data for lvtermine"> con columnas Day|Date|Time|Room.

Procesamiento en paralelo con ThreadPoolExecutor (max 5 workers).

Dependencias: requests, beautifulsoup4
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COURSES_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "courses.json"
OUTPUT_PATH  = Path(__file__).resolve().parents[1] / "data" / "raw" / "syllabi.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT = 10   # segundos por petición HTTP
MAX_WORKERS     = 5    # peticiones paralelas
RETRY_ATTEMPTS  = 2    # reintentos por curso en caso de error de red

# Mapa anchor → clave de sección
SECTION_MAP = {
    "lvbeschreibung1": "contents",
    "lvbeschreibung2": "learning_outcomes",
    "lvbeschreibung3": "attendance_requirements",
    "lvbeschreibung4": "teaching_methods",
    "lvbeschreibung5": "assessment",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers de limpieza de texto
# ---------------------------------------------------------------------------

def _clean_text(tag: Tag | None) -> str:
    """
    Extrae texto limpio de un tag BeautifulSoup:
    - Reemplaza <br> por salto de línea.
    - Elimina tags inline conservando el texto.
    - Colapsa espacios y líneas en blanco múltiples.
    """
    if tag is None:
        return ""
    # Reemplazar <br> por \n antes de extraer texto
    for br in tag.find_all("br"):
        br.replace_with("\n")
    text = tag.get_text(separator="\n")
    # Colapsar espacios horizontales, preservar saltos de línea significativos
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Eliminar líneas vacías consecutivas
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = line == ""
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Extractores de secciones y horarios
# ---------------------------------------------------------------------------

def _extract_section(soup: BeautifulSoup, anchor_name: str) -> str:
    """
    Localiza el anchor <a name="{anchor_name}">, sube al .panel-heading
    y devuelve el texto limpio del .panel-body hermano siguiente.
    Devuelve "" si la sección no existe en la página.
    """
    anchor = soup.find("a", {"name": anchor_name})
    if not anchor:
        return ""
    panel_heading = anchor.find_parent("div", class_="panel-heading")
    if not panel_heading:
        return ""
    panel_body = panel_heading.find_next_sibling("div", class_="panel-body")
    if not panel_body:
        return ""
    return _clean_text(panel_body)


def _extract_schedule(soup: BeautifulSoup) -> list[dict]:
    """
    Busca la tabla con summary="Data for lvtermine" (Day|Date|Time|Room)
    y devuelve una lista de dicts con las filas del <tbody>.
    """
    table = soup.find("table", {"summary": "Data for lvtermine"})
    if not table:
        # Fallback: cualquier tabla con esos cuatro headers
        for tbl in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            if {"Day", "Date", "Time", "Room"}.issubset(set(headers)):
                table = tbl
                break
    if not table:
        return []

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return []

    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) >= 4:
            rows.append({
                "day":  cells[0],
                "date": cells[1],
                "time": cells[2],
                "room": cells[3],
            })
    return rows


# ---------------------------------------------------------------------------
# Fetcher principal por curso
# ---------------------------------------------------------------------------

def _fetch_syllabus(course: dict) -> dict:
    """
    Descarga y parsea el syllabus de un único curso.
    Devuelve el dict enriquecido con 'sections' y 'schedule'.
    En caso de error HTTP o timeout, añade sections con status 'error'.
    """
    url  = course.get("syllabus_url", "")
    code = course.get("code", "?")
    name = course.get("name", "?")

    base = {
        "code":        code,
        "name":        name,
        "credits":     course.get("credits", ""),
        "type":        course.get("type", ""),
        "syllabus_url": url,
        "semester":    course.get("semester", ""),
    }
    empty_sections = {key: "" for key in SECTION_MAP.values()}

    if not url:
        log.warning("[%s] Sin syllabus_url, saltando.", code)
        return {**base, "sections": {**empty_sections, "status": "no_url"}, "schedule": []}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.info("[%s] GET %s (intento %d)", code, url, attempt)
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 404:
                log.warning("[%s] 404 — %s", code, url)
                return {**base, "sections": {**empty_sections, "status": "404"}, "schedule": []}

            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            sections: dict[str, str] = {}
            for anchor_name, section_key in SECTION_MAP.items():
                text = _extract_section(soup, anchor_name)
                sections[section_key] = text
                if text:
                    log.debug("[%s] ✓ sección '%s' (%d chars)", code, section_key, len(text))
                else:
                    log.debug("[%s] — sección '%s' vacía", code, section_key)

            schedule = _extract_schedule(soup)
            log.info(
                "[%s] OK — %d secciones con contenido | %d sesiones de horario",
                code,
                sum(1 for v in sections.values() if v),
                len(schedule),
            )
            return {**base, "sections": sections, "schedule": schedule}

        except requests.exceptions.Timeout:
            log.warning("[%s] Timeout en intento %d/%d", code, attempt, RETRY_ATTEMPTS)
        except requests.exceptions.ConnectionError as exc:
            log.warning("[%s] ConnectionError intento %d/%d: %s", code, attempt, RETRY_ATTEMPTS, exc)
        except requests.exceptions.HTTPError as exc:
            log.warning("[%s] HTTPError: %s", code, exc)
            return {**base, "sections": {**empty_sections, "status": "error"}, "schedule": []}

        if attempt < RETRY_ATTEMPTS:
            time.sleep(1.5)

    log.error("[%s] Falló tras %d intentos. URL: %s", code, RETRY_ATTEMPTS, url)
    return {**base, "sections": {**empty_sections, "status": "error"}, "schedule": []}


# ---------------------------------------------------------------------------
# Orquestador paralelo
# ---------------------------------------------------------------------------

def scrape_syllabi(courses: list[dict]) -> tuple[list[dict], int, int]:
    """
    Procesa todos los cursos en paralelo con hasta MAX_WORKERS workers.
    Devuelve (resultados_ordenados, n_ok, n_error).
    """
    results: list[dict] = [{}] * len(courses)
    n_ok = 0
    n_error = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_fetch_syllabus, course): idx
            for idx, course in enumerate(courses)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                log.error("Error inesperado en worker para índice %d: %s", idx, exc)
                c = courses[idx]
                result = {
                    "code": c.get("code", "?"),
                    "name": c.get("name", "?"),
                    "credits": c.get("credits", ""),
                    "type": c.get("type", ""),
                    "syllabus_url": c.get("syllabus_url", ""),
                    "semester": c.get("semester", ""),
                    "sections": {k: "" for k in SECTION_MAP.values()} | {"status": "error"},
                    "schedule": [],
                }

            status = result.get("sections", {}).get("status", "ok")
            if status in ("error", "404", "no_url"):
                n_error += 1
            else:
                n_ok += 1

            results[idx] = result

    return results, n_ok, n_error


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not COURSES_PATH.exists():
        log.error("No se encontró %s — ejecuta phase1_catalog.py primero.", COURSES_PATH)
        return

    courses = json.loads(COURSES_PATH.read_text(encoding="utf-8"))
    log.info("Cargados %d cursos desde %s", len(courses), COURSES_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results, n_ok, n_error = scrape_syllabi(courses)

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*52}")
    print(f"  Syllabi extraídos OK:    {n_ok}")
    print(f"  Syllabi con error:       {n_error}")
    print(f"  Total procesados:        {len(results)}")
    print(f"  Resultado en:            {OUTPUT_PATH}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
