"""
phase2_syllabi_s26.py
---------------------
Scraper de fase 2 para Summer 2026: descarga los syllabi de los cursos generados
por phase1_catalog_s26.py y los guarda en data/raw/syllabi_s26.json.

Diferencias respecto a phase2_syllabi.py:
  - Lee data/raw/courses_s26.json  (no sobreescribe los de Winter 2025)
  - Escribe data/raw/syllabi_s26.json
  - Añade campo "schedule_pending": true cuando el horario aún no está publicado
    (schedule vacío), para identificar asignaturas que hay que revisar más adelante.
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

COURSES_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "courses_s26.json"
OUTPUT_PATH  = Path(__file__).resolve().parents[1] / "data" / "raw" / "syllabi_s26.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
REQUEST_TIMEOUT = 10
MAX_WORKERS     = 5
RETRY_ATTEMPTS  = 2

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
# Helpers de limpieza
# ---------------------------------------------------------------------------

def _clean_text(tag: Tag | None) -> str:
    if tag is None:
        return ""
    for br in tag.find_all("br"):
        br.replace_with("\n")
    text = tag.get_text(separator="\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
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
# Extractores
# ---------------------------------------------------------------------------

def _extract_section(soup: BeautifulSoup, anchor_name: str) -> str:
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
    table = soup.find("table", {"summary": "Data for lvtermine"})
    if not table:
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
# Fetcher por curso
# ---------------------------------------------------------------------------

def _fetch_syllabus(course: dict) -> dict:
    url  = course.get("syllabus_url", "")
    code = course.get("code", "?")
    name = course.get("name", "?")

    base = {
        "code":         code,
        "name":         name,
        "credits":      course.get("credits", ""),
        "type":         course.get("type", ""),
        "syllabus_url": url,
        "semester":     course.get("semester", ""),
    }
    empty_sections = {key: "" for key in SECTION_MAP.values()}

    if not url:
        log.warning("[%s] Sin syllabus_url, saltando.", code)
        return {
            **base,
            "sections": {**empty_sections, "status": "no_url"},
            "schedule": [],
            "schedule_pending": True,
        }

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.info("[%s] GET %s (intento %d)", code, url, attempt)
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 404:
                log.warning("[%s] 404 — %s", code, url)
                return {
                    **base,
                    "sections": {**empty_sections, "status": "404"},
                    "schedule": [],
                    "schedule_pending": True,
                }

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
            schedule_pending = len(schedule) == 0

            if schedule_pending:
                log.info("[%s] ⚠ Horario aún no publicado (schedule_pending=True).", code)
            else:
                log.info(
                    "[%s] OK — %d secciones con contenido | %d sesiones de horario",
                    code,
                    sum(1 for v in sections.values() if v),
                    len(schedule),
                )

            return {
                **base,
                "sections": sections,
                "schedule": schedule,
                "schedule_pending": schedule_pending,
            }

        except requests.exceptions.Timeout:
            log.warning("[%s] Timeout en intento %d/%d", code, attempt, RETRY_ATTEMPTS)
        except requests.exceptions.ConnectionError as exc:
            log.warning("[%s] ConnectionError intento %d/%d: %s", code, attempt, RETRY_ATTEMPTS, exc)
        except requests.exceptions.HTTPError as exc:
            log.warning("[%s] HTTPError: %s", code, exc)
            return {
                **base,
                "sections": {**empty_sections, "status": "error"},
                "schedule": [],
                "schedule_pending": True,
            }

        if attempt < RETRY_ATTEMPTS:
            time.sleep(1.5)

    log.error("[%s] Falló tras %d intentos. URL: %s", code, RETRY_ATTEMPTS, url)
    return {
        **base,
        "sections": {**empty_sections, "status": "error"},
        "schedule": [],
        "schedule_pending": True,
    }


# ---------------------------------------------------------------------------
# Orquestador paralelo
# ---------------------------------------------------------------------------

def scrape_syllabi(courses: list[dict]) -> tuple[list[dict], int, int, int]:
    results: list[dict] = [{}] * len(courses)
    n_ok      = 0
    n_error   = 0
    n_pending = 0

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
                    "code":             c.get("code", "?"),
                    "name":             c.get("name", "?"),
                    "credits":          c.get("credits", ""),
                    "type":             c.get("type", ""),
                    "syllabus_url":     c.get("syllabus_url", ""),
                    "semester":         c.get("semester", ""),
                    "sections":         {k: "" for k in SECTION_MAP.values()} | {"status": "error"},
                    "schedule":         [],
                    "schedule_pending": True,
                }

            status = result.get("sections", {}).get("status", "ok")
            if status in ("error", "404", "no_url"):
                n_error += 1
            else:
                n_ok += 1
                if result.get("schedule_pending"):
                    n_pending += 1

            results[idx] = result

    return results, n_ok, n_error, n_pending


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not COURSES_PATH.exists():
        log.error("No se encontró %s — ejecuta phase1_catalog_s26.py primero.", COURSES_PATH)
        return

    courses = json.loads(COURSES_PATH.read_text(encoding="utf-8"))
    log.info("Cargados %d cursos desde %s", len(courses), COURSES_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results, n_ok, n_error, n_pending = scrape_syllabi(courses)

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*58}")
    print(f"  Syllabi extraídos OK:        {n_ok}")
    print(f"    ⚠ Sin horario publicado:   {n_pending}  (schedule_pending)")
    print(f"  Syllabi con error:           {n_error}")
    print(f"  Total procesados:            {len(results)}")
    print(f"  Resultado en:                {OUTPUT_PATH}")
    print(f"{'='*58}")

    if n_pending:
        pending_codes = [r["code"] for r in results if r.get("schedule_pending") and
                         r.get("sections", {}).get("status", "ok") == "ok"]
        print(f"\n  Asignaturas sin horario ({len(pending_codes)}):")
        for code in pending_codes:
            name = next((r["name"] for r in results if r["code"] == code), "")
            print(f"    [{code}] {name}")


if __name__ == "__main__":
    main()
