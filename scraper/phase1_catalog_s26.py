"""
phase1_catalog_s26.py
---------------------
Scraper de fase 1 para Summer 2026: extrae el catálogo de asignaturas WU Vienna
del semestre de verano 2026. Genera data/raw/courses_s26.json sin sobreescribir
los datos de Winter 2025 (data/raw/courses.json).

Idéntico a phase1_catalog.py salvo: CATALOG_URL, SEMESTER_LABEL y OUTPUT_PATH.
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
    "course-catalog/filter/97102/1,5/3/0//"
)

OUTPUT_PATH    = Path(__file__).resolve().parents[1] / "data" / "raw" / "courses_s26.json"
SEMESTER_LABEL = "Summer 2026"

PAGE_LOAD_TIMEOUT   = 15
MODAL_TIMEOUT       = 8
BETWEEN_CLICKS_DELAY = 0.5

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
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1400,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = uc.Chrome(options=options, version_main=149)
    driver.implicitly_wait(1)
    return driver


# ---------------------------------------------------------------------------
# Helpers de extracción
# ---------------------------------------------------------------------------

def _dismiss_cookie_banner(driver: uc.Chrome) -> None:
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
    parts = [p.strip() for p in location_text.split("|")]
    code        = parts[0] if len(parts) > 0 else ""
    level       = parts[1] if len(parts) > 1 else ""
    language    = parts[2] if len(parts) > 2 else ""
    course_type = parts[4] if len(parts) > 4 else (parts[3] if len(parts) > 3 else "")
    return code, level, language, course_type


def _dl_value(modal_el, label: str) -> str:
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
    try:
        active = driver.find_element(By.CSS_SELECTOR, "div.pagination-page li.active span")
        return int(active.text.strip())
    except (NoSuchElementException, ValueError):
        return -1


def _next_page_available(driver: uc.Chrome) -> bool:
    try:
        span = driver.find_element(By.CSS_SELECTOR, "div.pagination-page span.next")
        li = driver.execute_script("return arguments[0].parentElement;", span)
        return "disabled" not in (li.get_attribute("class") or "")
    except NoSuchElementException:
        return False


def _click_next_page(driver: uc.Chrome) -> bool:
    try:
        pg_before = _current_page_num(driver)
        span_next = driver.find_element(By.CSS_SELECTOR, "div.pagination-page span.next")
        ActionChains(driver).move_to_element(span_next).click().perform()

        deadline = time.time() + 6
        while time.time() < deadline:
            time.sleep(0.2)
            if _current_page_num(driver) == pg_before + 1:
                inner = time.time() + 3
                while time.time() < inner:
                    visible = [
                        e for e in driver.find_elements(By.CSS_SELECTOR, "div.map__entry")
                        if e.is_displayed()
                    ]
                    if len(visible) > 0:
                        return True
                    time.sleep(0.15)
                return True
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
    extracted = 0
    skipped   = 0

    visible_entries = [
        e for e in driver.find_elements(By.CSS_SELECTOR, "div.map__entry")
        if e.is_displayed()
    ]
    log.info("── Página %d: %d entradas visibles ──", page_num, len(visible_entries))

    for pos, entry in enumerate(visible_entries):
        idx_label = global_idx + pos + 1
        try:
            name     = entry.find_element(By.CSS_SELECTOR, ".map__entry-title").text.strip()
            location = entry.find_element(By.CSS_SELECTOR, ".map__entry-location").text.strip()
        except (NoSuchElementException, StaleElementReferenceException):
            log.warning("[%d] No se pudo leer la entrada, saltando.", idx_label)
            skipped += 1
            continue

        code, level, language, course_type = _parse_location(location)
        log.info("[%d] %s | código: %s", idx_label, name[:55], code)

        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", entry)
            time.sleep(0.15)
            driver.execute_script("arguments[0].click();", entry)

            modal = modal_wait.until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, "div.map__modal.in"))
            )
            credits     = _dl_value(modal, "ECTS credits")
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
            "code":        code,
            "name":        name,
            "credits":     credits,
            "type":        course_type,
            "syllabus_url": syllabus_url,
            "semester":    SEMESTER_LABEL,
        })
        log.info("  ✓ %s ECTS | tipo: %s | %s", credits, course_type, syllabus_url)
        extracted += 1
        time.sleep(BETWEEN_CLICKS_DELAY)

    return extracted, skipped


# ---------------------------------------------------------------------------
# Scraping principal
# ---------------------------------------------------------------------------

def scrape_catalog() -> tuple[list[dict], int]:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    courses: list[dict] = []
    total_skipped = 0

    driver     = build_driver()
    modal_wait = WebDriverWait(driver, MODAL_TIMEOUT)

    try:
        log.info("Abriendo catálogo WU Summer 2026…")
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
        log.info("Entradas totales en DOM: %d", total_in_dom)

        global_idx = 0
        page_num   = 1

        while True:
            extracted, skipped = _process_visible_entries(
                driver, modal_wait, courses, page_num, global_idx
            )
            global_idx    += extracted + skipped
            total_skipped += skipped

            if not _next_page_available(driver):
                log.info("No hay más páginas. Scraping completado.")
                break

            log.info("Avanzando a página %d…", page_num + 1)
            if not _click_next_page(driver):
                log.warning("No se pudo cambiar a página %d. Abortando.", page_num + 1)
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
    print(f"  Semestre:              {SEMESTER_LABEL}")
    print(f"  Asignaturas extraídas: {len(courses)}")
    print(f"  Asignaturas saltadas:  {skipped}")
    print(f"  JSON guardado en:      {OUTPUT_PATH}")
    print(f"{'='*52}")


if __name__ == "__main__":
    main()
