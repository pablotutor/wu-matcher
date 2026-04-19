"""
phase1_catalog.py
-----------------
Scraper de fase 1: extrae el catálogo completo de asignaturas de la WU Vienna
(Wirtschaftsuniversität Wien) desde su buscador de cursos para estudiantes de intercambio.

Estructura real del DOM (confirmada por inspección):

  Cada entrada de la lista:
    <div class="map__entry" id="{internal_id}">
      <div class="map__entry-title">{Nombre}</div>
      <div class="map__entry-location">{code} | {level} | {lang} | {dept} | {type}</div>
    </div>

  Modal Bootstrap (clase .in cuando está visible):
    <div class="modal map__modal fade in">
      <span class="map__modal-title">{code} {Nombre}</span>
      <dl class="dl-horizontal">
        <dt>ECTS credits:</dt> <dd>{n}</dd>
        <dt>Course type:</dt>  <dd>{PI/SE/...}</dd>
        <dt>Syllabus:</dt>     <dd><a href="...">url</a></dd>
        ...
      </dl>
    </div>

Estrategia:
  - Datos básicos (nombre, código, tipo) → parsear directamente del texto de cada entrada.
  - ECTS y syllabus_url → abrir modal (Bootstrap, esperar clase .in) y leer el <dl>.
  - Cerrar modal con el botón .map__modal-close (data-dismiss="modal").

Dependencias: undetected-chromedriver, selenium
"""

import json
import re
import time
import logging
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
    StaleElementReferenceException,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CATALOG_URL = (
    "https://www.wu.ac.at/en/incoming-students/exchange-semester/academics/"
    "course-catalog/filter/92389/1,5/3/0//"
)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "courses.json"
SEMESTER_LABEL = "Winter 2025"

PAGE_LOAD_TIMEOUT = 15    # segundos para que cargue div.map__entry-list
MODAL_TIMEOUT = 8         # segundos para que Bootstrap añada .in al modal
BETWEEN_CLICKS_DELAY = 0.5  # pausa entre asignaturas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Driver setup
# ---------------------------------------------------------------------------

def build_driver() -> uc.Chrome:
    """Lanza Chrome no-detectable. Sin headless para mayor estabilidad con Bootstrap."""
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1400,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options)
    driver.implicitly_wait(1)
    return driver


# ---------------------------------------------------------------------------
# Helpers de extracción
# ---------------------------------------------------------------------------

def _dismiss_cookie_banner(driver: uc.Chrome) -> None:
    """Cierra el banner de cookies si aparece, para que no bloquee los clicks."""
    for sel in [
        "button.cookie-notice-accept",
        "button[data-action='accept']",
        ".cookie-notice-modal .close",
        "#cookie-notice button",
        "button.js-accept-all",
    ]:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
                log.info("Banner de cookies cerrado.")
                time.sleep(0.5)
                return
        except (NoSuchElementException, ElementClickInterceptedException):
            pass


def _parse_location(location_text: str) -> tuple[str, str, str, str]:
    """
    Parsea el texto de .map__entry-location con formato:
      '{code} | {level} | {language} | {department} | {type}'

    Devuelve (code, level, language, course_type).
    """
    parts = [p.strip() for p in location_text.split("|")]
    code = parts[0] if len(parts) > 0 else ""
    level = parts[1] if len(parts) > 1 else ""
    language = parts[2] if len(parts) > 2 else ""
    course_type = parts[4] if len(parts) > 4 else (parts[3] if len(parts) > 3 else "")
    return code, level, language, course_type


def _dl_value(modal_el, label: str) -> str:
    """
    Busca en el <dl class="dl-horizontal"> el <dd> que sigue al <dt>
    cuyo texto contenga `label` (case-insensitive). Devuelve "" si no lo encuentra.
    """
    try:
        dts = modal_el.find_elements(By.CSS_SELECTOR, "dl.dl-horizontal dt")
        for dt in dts:
            if label.lower() in dt.text.strip().lower():
                dd = dt.find_element(By.XPATH, "following-sibling::dd[1]")
                return dd.text.strip()
    except (NoSuchElementException, StaleElementReferenceException):
        pass
    return ""


def _syllabus_href(modal_el) -> str:
    """Devuelve el href del enlace en el <dd> que sigue a 'Syllabus:'."""
    try:
        dts = modal_el.find_elements(By.CSS_SELECTOR, "dl.dl-horizontal dt")
        for dt in dts:
            if "syllabus" in dt.text.strip().lower():
                dd = dt.find_element(By.XPATH, "following-sibling::dd[1]")
                a = dd.find_element(By.TAG_NAME, "a")
                return a.get_attribute("href") or ""
    except (NoSuchElementException, StaleElementReferenceException):
        pass
    return ""


def _close_modal(driver: uc.Chrome) -> None:
    """Cierra el modal Bootstrap activo con su botón de cierre nativo."""
    try:
        btn = driver.find_element(By.CSS_SELECTOR, "button.map__modal-close")
        driver.execute_script("arguments[0].click();", btn)
        WebDriverWait(driver, 3).until_not(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.map__modal.in"))
        )
    except Exception:
        driver.execute_script(
            "var m = document.querySelector('.map__modal'); "
            "if(m){ m.classList.remove('in'); m.style.display='none'; }"
            "var bd = document.querySelector('.modal-backdrop'); if(bd) bd.remove();"
            "document.body.classList.remove('modal-open');"
        )


def _current_page_num(driver: uc.Chrome) -> int:
    """Devuelve el número de página activa según el paginador simple-pagination."""
    try:
        active = driver.find_element(By.CSS_SELECTOR, "div.pagination-page li.active span")
        return int(active.text.strip())
    except (NoSuchElementException, ValueError):
        return -1


def _next_page_available(driver: uc.Chrome) -> bool:
    """True si span.next existe y su <li> padre no tiene clase 'disabled'."""
    try:
        span = driver.find_element(By.CSS_SELECTOR, "div.pagination-page span.next")
        li = driver.execute_script("return arguments[0].parentElement;", span)
        return "disabled" not in (li.get_attribute("class") or "")
    except NoSuchElementException:
        return False


def _click_next_page(driver: uc.Chrome) -> bool:
    """
    Avanza a la siguiente página con ActionChains sobre span.next
    (único método que activa el plugin simple-pagination).
    Confirma el cambio esperando que el número de página activo incremente.
    Devuelve True si el cambio se confirmó, False en caso de timeout/error.
    """
    try:
        pg_before = _current_page_num(driver)
        span_next = driver.find_element(By.CSS_SELECTOR, "div.pagination-page span.next")
        ActionChains(driver).move_to_element(span_next).click().perform()

        deadline = time.time() + 6
        while time.time() < deadline:
            time.sleep(0.2)
            if _current_page_num(driver) == pg_before + 1:
                # Esperar a que las 20 entradas de la nueva página estén visibles
                inner = time.time() + 3
                while time.time() < inner:
                    visible = [
                        e for e in driver.find_elements(By.CSS_SELECTOR, "div.map__entry")
                        if e.is_displayed()
                    ]
                    if len(visible) > 0:
                        return True
                    time.sleep(0.15)
                return True  # página cambió aunque no haya exactamente 20
        return False
    except Exception as exc:
        log.warning("Error al cambiar de página: %s", exc)
        return False


def _process_visible_entries(
    driver: uc.Chrome,
    modal_wait: WebDriverWait,
    courses: list,
    page_num: int,
    global_idx: int,
) -> tuple[int, int]:
    """
    Procesa todas las entradas visibles de la página actual.
    Devuelve (n_extraídas, n_saltadas).
    """
    extracted = 0
    skipped = 0

    visible_entries = [
        e for e in driver.find_elements(By.CSS_SELECTOR, "div.map__entry")
        if e.is_displayed()
    ]
    log.info("── Página %d: %d entradas visibles ──", page_num, len(visible_entries))

    for pos, entry in enumerate(visible_entries):
        idx_label = global_idx + pos + 1
        try:
            name = entry.find_element(By.CSS_SELECTOR, ".map__entry-title").text.strip()
            location = entry.find_element(By.CSS_SELECTOR, ".map__entry-location").text.strip()
        except (NoSuchElementException, StaleElementReferenceException):
            log.warning("[%d] No se pudo leer la entrada, saltando.", idx_label)
            skipped += 1
            continue

        code, level, language, course_type = _parse_location(location)
        log.info("[%d] %s | código: %s", idx_label, name[:55], code)

        # Click para abrir modal (ECTS + syllabus URL)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", entry)
            time.sleep(0.15)
            driver.execute_script("arguments[0].click();", entry)

            modal = modal_wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.map__modal.in"))
            )
            credits = _dl_value(modal, "ECTS credits")
            syllabus_url = _syllabus_href(modal)
            _close_modal(driver)

        except TimeoutException:
            log.warning("  Modal no apareció para '%s', saltando.", name[:40])
            skipped += 1
            continue
        except StaleElementReferenceException:
            log.warning("  StaleElement en '%s', saltando.", name[:40])
            skipped += 1
            continue

        courses.append({
            "code": code,
            "name": name,
            "credits": credits,
            "type": course_type,
            "syllabus_url": syllabus_url,
            "semester": SEMESTER_LABEL,
        })
        log.info("  ✓ %s ECTS | tipo: %s | %s", credits, course_type, syllabus_url)
        extracted += 1
        time.sleep(BETWEEN_CLICKS_DELAY)

    return extracted, skipped


# ---------------------------------------------------------------------------
# Lógica principal de scraping
# ---------------------------------------------------------------------------

def scrape_catalog() -> tuple[list[dict], int]:
    """
    Abre el catálogo WU e itera por todas las páginas del paginador
    simple-pagination (20 entradas visibles por página, ~17 páginas).

    Por cada entrada:
      - Del HTML de la entrada: nombre, código, tipo.
      - Del modal Bootstrap (.map__modal.in): ECTS credits y syllabus_url.

    Devuelve (lista_de_cursos, n_saltados).
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    courses: list[dict] = []
    total_skipped = 0

    driver = build_driver()
    modal_wait = WebDriverWait(driver, MODAL_TIMEOUT)

    try:
        log.info("Abriendo catálogo WU…")
        driver.get(CATALOG_URL)

        log.info("Esperando div.map__entry-list (max %ds)…", PAGE_LOAD_TIMEOUT)
        try:
            WebDriverWait(driver, PAGE_LOAD_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.map__entry-list"))
            )
        except TimeoutException:
            log.error("La lista de asignaturas no cargó. Abortando.")
            return courses, total_skipped

        _dismiss_cookie_banner(driver)

        total_in_dom = len(driver.find_elements(By.CSS_SELECTOR, "div.map__entry"))
        log.info("Entradas totales en DOM: %d (distribuidas en varias páginas)", total_in_dom)

        global_idx = 0
        page_num = 1

        while True:
            extracted, skipped = _process_visible_entries(
                driver, modal_wait, courses, page_num, global_idx
            )
            global_idx += extracted + skipped
            total_skipped += skipped

            if not _next_page_available(driver):
                log.info("No hay más páginas. Scraping completado.")
                break

            log.info("Avanzando a página %d…", page_num + 1)
            if not _click_next_page(driver):
                log.warning("No se pudo cambiar a página %d. Abortando paginación.", page_num + 1)
                break

            page_num += 1

    finally:
        driver.quit()
        log.info("Navegador cerrado.")

    return courses, total_skipped


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    courses, skipped = scrape_catalog()

    OUTPUT_PATH.write_text(
        json.dumps(courses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*52}")
    print(f"  Asignaturas extraídas: {len(courses)}")
    print(f"  Asignaturas saltadas:  {skipped}")
    print(f"  JSON guardado en:      {OUTPUT_PATH}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
