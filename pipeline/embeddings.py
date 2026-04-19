"""
embeddings.py
-------------
Genera embeddings para todos los chunks procesados y los persiste en ChromaDB.

Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384 dims)
Colección Chroma: "wu_syllabi"
  - ID:       chunk["id"]
  - Document: chunk["text"]
  - Embedding: vector 384-dim
  - Metadata: code, name, section, credits, type, schedule (JSON string)

Procesamiento:
  - CPU (compatible con Apple M1)
  - Batch size: 32 chunks por encode()
  - Checkpoint: upsert a Chroma cada 100 chunks

Dependencias: sentence-transformers, chromadb
"""

import json
import logging
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNKS_PATH  = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
CHROMA_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed" / "chroma_db"

MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "wu_syllabi"
EMBED_DIM       = 384
BATCH_SIZE      = 32   # chunks por llamada a model.encode()
CHECKPOINT_EVERY = 100  # upsert a Chroma cada N chunks

VALIDATION_QUERY = "machine learning"
VALIDATION_TOP_K = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks(chunks_path: Path) -> tuple[list[str], list[str], list[dict]]:
    """
    Lee chunks.json y devuelve tres listas paralelas:
      ids, texts, metadatas
    El campo 'schedule' se serializa como JSON string (Chroma no admite listas en metadata).
    """
    data = json.loads(chunks_path.read_text(encoding="utf-8"))

    ids:       list[str]  = []
    texts:     list[str]  = []
    metas:     list[dict] = []

    for course in data:
        code     = course.get("code", "")
        name     = course.get("name", "")
        metadata = course.get("metadata", {})
        credits  = metadata.get("credits", "")
        ctype    = metadata.get("type", "")
        schedule = json.dumps(metadata.get("schedule", []), ensure_ascii=False)

        for chunk in course.get("chunks", []):
            ids.append(chunk["id"])
            texts.append(chunk["text"])
            metas.append({
                "code":     code,
                "name":     name,
                "section":  chunk["section"],
                "credits":  credits,
                "type":     ctype,
                "schedule": schedule,
            })

    return ids, texts, metas


def _get_collection(chroma_dir: Path) -> chromadb.Collection:
    """Crea o abre la colección persistente 'wu_syllabi'."""
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Función pública: embed_query
# ---------------------------------------------------------------------------

def embed_query(text: str, model: SentenceTransformer | None = None) -> list[float]:
    """
    Vectoriza una consulta en tiempo real.
    Si no se pasa `model`, carga el modelo desde disco.
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    return model.encode(text, convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def generate_embeddings() -> None:
    if not CHUNKS_PATH.exists():
        log.error("No se encontró %s — ejecuta chunker.py primero.", CHUNKS_PATH)
        return

    log.info("Cargando chunks desde %s …", CHUNKS_PATH)
    ids, texts, metas = _load_chunks(CHUNKS_PATH)
    total = len(ids)
    log.info("Total de chunks a procesar: %d", total)

    log.info("Cargando modelo %s en CPU …", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    collection = _get_collection(CHROMA_DIR)

    # Determinar qué IDs ya existen para no re-embedar (reanudación tras crash)
    existing_ids: set[str] = set()
    try:
        existing = collection.get(include=[])
        existing_ids = set(existing["ids"])
        if existing_ids:
            log.info("Encontrados %d chunks ya indexados — se saltarán.", len(existing_ids))
    except Exception:
        pass

    pending_ids   = [i for i in ids   if i not in existing_ids]
    pending_texts = [texts[ids.index(i)] for i in pending_ids]
    pending_metas = [metas[ids.index(i)] for i in pending_ids]
    pending_total = len(pending_ids)

    if pending_total == 0:
        log.info("Todos los chunks ya están indexados. Nada que hacer.")
    else:
        log.info("Procesando %d chunks nuevos …", pending_total)

    t0 = time.time()
    n_done = 0

    # Buffer para checkpoint
    buf_ids:   list[str]       = []
    buf_texts: list[str]       = []
    buf_metas: list[dict]      = []
    buf_embs:  list[list[float]] = []

    for batch_start in range(0, pending_total, BATCH_SIZE):
        batch_end   = min(batch_start + BATCH_SIZE, pending_total)
        b_ids   = pending_ids[batch_start:batch_end]
        b_texts = pending_texts[batch_start:batch_end]
        b_metas = pending_metas[batch_start:batch_end]

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
        log.info("Embeddings generados: %d/%d", len(existing_ids) + n_done, total)

        # Checkpoint cada CHECKPOINT_EVERY chunks
        if len(buf_ids) >= CHECKPOINT_EVERY:
            collection.upsert(
                ids=buf_ids,
                documents=buf_texts,
                embeddings=buf_embs,
                metadatas=buf_metas,
            )
            log.info("  → Checkpoint: %d chunks guardados en Chroma.", len(buf_ids))
            buf_ids, buf_texts, buf_metas, buf_embs = [], [], [], []

    # Flush del buffer restante
    if buf_ids:
        collection.upsert(
            ids=buf_ids,
            documents=buf_texts,
            embeddings=buf_embs,
            metadatas=buf_metas,
        )
        log.info("  → Flush final: %d chunks guardados en Chroma.", len(buf_ids))

    elapsed = time.time() - t0

    # ---------------------------------------------------------------------------
    # Validación
    # ---------------------------------------------------------------------------
    log.info("Ejecutando búsqueda de validación: '%s' …", VALIDATION_QUERY)
    query_emb = embed_query(VALIDATION_QUERY, model)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=VALIDATION_TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    # ---------------------------------------------------------------------------
    # Resumen
    # ---------------------------------------------------------------------------
    print(f"\n{'='*58}")
    print(f"  Embeddings generados: {len(existing_ids) + n_done}/{total}")
    print(f"  Tiempo total:         {elapsed:.1f}s")
    print(f"  Colección Chroma:     {COLLECTION_NAME}  ({collection.count()} docs)")
    print(f"  DB guardada en:       {CHROMA_DIR}")
    print(f"\n  Búsqueda de prueba: \"{VALIDATION_QUERY}\"")
    print(f"  {'─'*54}")

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), 1):
        similarity = round(1 - dist, 4)
        snippet = doc[:120].replace("\n", " ")
        print(f"  [{rank}] {meta['code']} | {meta['name'][:40]}")
        print(f"      Sección: {meta['section']}  |  Similitud: {similarity}")
        print(f"      \"{snippet}…\"")
        print()

    print(f"{'='*58}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_embeddings()
