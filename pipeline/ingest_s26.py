"""
ingest_s26.py
-------------
Orquestador de ingesta para Summer 2026.

Flujo:
  1. Lee data/raw/syllabi_s26.json
  2. Chunkea los nuevos syllabi (reutiliza process_syllabus de chunker.py)
  3. Guarda chunks nuevos en data/processed/chunks_s26.json
  4. Fusiona chunks_s26.json con chunks.json (sin duplicar por código)
     → actualiza chunks.json para que HybridRetriever recoja los nuevos cursos
  5. Fuerza reconstrucción de la colección 'wu_syllabi_courses' en ChromaDB

Preserva:
  - data/raw/courses.json      (Winter 2025, intacto)
  - data/raw/syllabi.json      (Winter 2025, intacto)
  - Chunks de Winter 2025 dentro de chunks.json (no se borran)

Uso:
  python pipeline/ingest_s26.py
"""

import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from pipeline.chunker import process_syllabus

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYLLABI_S26_PATH = _PROJECT_ROOT / "data" / "raw" / "syllabi_s26.json"
CHUNKS_S26_PATH  = _PROJECT_ROOT / "data" / "processed" / "chunks_s26.json"
CHUNKS_PATH      = _PROJECT_ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR       = _PROJECT_ROOT / "data" / "processed" / "chroma_db"

MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "wu_syllabi_courses"
BATCH_SIZE      = 32
RELEVANT_SECTIONS = {"contents", "learning_outcomes"}

import re
_PREFIX_RE = re.compile(r"^\[[A-Z_]+\]\s+\S+\s+\|\s*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paso 1 & 2: Chunking de syllabi_s26.json
# ---------------------------------------------------------------------------

def chunk_s26_syllabi() -> list[dict]:
    if not SYLLABI_S26_PATH.exists():
        log.error("No se encontró %s — ejecuta phase2_syllabi_s26.py primero.", SYLLABI_S26_PATH)
        sys.exit(1)

    syllabi = json.loads(SYLLABI_S26_PATH.read_text(encoding="utf-8"))
    log.info("Cargados %d syllabi de Summer 2026.", len(syllabi))

    processed: list[dict] = []
    total_chunks = 0

    for syl in syllabi:
        status = syl.get("sections", {}).get("status", "")
        if status in ("error", "404", "no_url"):
            log.warning("[%s] Saltado por status '%s'.", syl.get("code", "?"), status)
            continue
        doc = process_syllabus(syl)
        processed.append(doc)
        total_chunks += len(doc["chunks"])

    CHUNKS_S26_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_S26_PATH.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("chunks_s26.json: %d asignaturas | %d chunks.", len(processed), total_chunks)
    return processed


# ---------------------------------------------------------------------------
# Paso 3: Fusión en chunks.json
# ---------------------------------------------------------------------------

def merge_into_chunks(new_courses: list[dict]) -> list[dict]:
    """
    Fusiona new_courses en chunks.json.
    Si un curso (mismo código) ya existe, lo reemplaza (actualización de semestre).
    Si es nuevo, lo añade.
    Devuelve el corpus completo resultante.
    """
    if not CHUNKS_PATH.exists():
        log.error("No se encontró %s.", CHUNKS_PATH)
        sys.exit(1)

    existing = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    existing_by_code = {c["code"]: i for i, c in enumerate(existing)}

    added   = 0
    updated = 0

    for course in new_courses:
        code = course["code"]
        if code in existing_by_code:
            existing[existing_by_code[code]] = course
            updated += 1
        else:
            existing.append(course)
            added += 1

    CHUNKS_PATH.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("chunks.json actualizado: %d añadidos | %d actualizados | %d total.",
             added, updated, len(existing))
    return existing


# ---------------------------------------------------------------------------
# Paso 4: Reconstruir wu_syllabi_courses en ChromaDB
# ---------------------------------------------------------------------------

def _build_aggregated_corpus(all_chunks: list[dict]) -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metas = [], [], []

    for course in all_chunks:
        code     = course.get("code", "")
        name     = course.get("name", "")
        meta_c   = course.get("metadata", {})
        credits  = meta_c.get("credits", "")
        ctype    = meta_c.get("type", "")
        schedule = meta_c.get("schedule", [])

        relevant_parts: list[str] = []
        all_chunk_ids: list[str]  = []

        for chunk in course.get("chunks", []):
            all_chunk_ids.append(chunk["id"])
            if chunk["section"] not in RELEVANT_SECTIONS:
                continue
            clean_text = _PREFIX_RE.sub("", chunk["text"]).strip()
            if clean_text:
                relevant_parts.append(clean_text)

        if not relevant_parts:
            continue

        ids.append(code)
        texts.append("\n\n".join(relevant_parts))
        metas.append({
            "code":       code,
            "name":       name,
            "credits":    credits,
            "type":       ctype,
            "schedule":   json.dumps(schedule, ensure_ascii=False),
            "all_chunks": json.dumps(all_chunk_ids, ensure_ascii=False),
        })

    return ids, texts, metas


def rebuild_chroma(all_chunks: list[dict]) -> None:
    log.info("Cargando modelo de embeddings (%s) …", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    ids, texts, metas = _build_aggregated_corpus(all_chunks)
    log.info("Corpus total: %d asignaturas.", len(ids))

    # Borrar colección existente para reconstruir con el corpus completo
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        log.info("Eliminando colección antigua '%s' para reconstruir.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    t0     = time.time()
    n_done = 0

    buf_ids:   list[str]         = []
    buf_texts: list[str]         = []
    buf_metas: list[dict]        = []
    buf_embs:  list[list[float]] = []

    for i in range(0, len(ids), BATCH_SIZE):
        b_ids   = ids[i:i + BATCH_SIZE]
        b_texts = texts[i:i + BATCH_SIZE]
        b_metas = metas[i:i + BATCH_SIZE]

        embeddings = model.encode(
            b_texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

        buf_ids.extend(b_ids)
        buf_texts.extend(b_texts)
        buf_metas.extend(b_metas)
        buf_embs.extend(embeddings)
        n_done += len(b_ids)
        log.info("Embeddings: %d/%d", n_done, len(ids))

        if len(buf_ids) >= 100:
            collection.upsert(
                ids=buf_ids,
                documents=buf_texts,
                embeddings=buf_embs,
                metadatas=buf_metas,
            )
            log.info("  → Checkpoint: %d docs guardados.", len(buf_ids))
            buf_ids, buf_texts, buf_metas, buf_embs = [], [], [], []

    if buf_ids:
        collection.upsert(
            ids=buf_ids,
            documents=buf_texts,
            embeddings=buf_embs,
            metadatas=buf_metas,
        )
        log.info("  → Flush final: %d docs.", len(buf_ids))

    elapsed = time.time() - t0
    log.info("Reconstrucción completada en %.1fs.", elapsed)

    print(f"\n{'='*58}")
    print(f"  Colección:          {COLLECTION_NAME}")
    print(f"  Documentos totales: {collection.count()}")
    print(f"  Tiempo:             {elapsed:.1f}s")
    print(f"  DB en:              {CHROMA_DIR}")
    print(f"{'='*58}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Ingesta Summer 2026 ===")

    log.info("--- Paso 1/3: Chunking de syllabi_s26.json ---")
    new_chunks = chunk_s26_syllabi()

    log.info("--- Paso 2/3: Fusión en chunks.json ---")
    all_chunks = merge_into_chunks(new_chunks)

    log.info("--- Paso 3/3: Reconstrucción de ChromaDB ---")
    rebuild_chroma(all_chunks)

    log.info("=== Ingesta completada ===")


if __name__ == "__main__":
    main()
