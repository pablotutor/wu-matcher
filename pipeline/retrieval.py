"""
retrieval.py
------------
Búsqueda híbrida semántica + BM25 con Reciprocal Rank Fusion (RRF).

Arquitectura: UN documento por asignatura WU (321 docs totales).
Cada documento agrega solo las secciones relevantes:
  CONTENTS + LEARNING_OUTCOMES  (se descartan ASSESSMENT, ATTENDANCE, TEACHING_METHODS)

Flujo:
  1. Al inicializar, construye (si no existe) la colección "wu_syllabi_courses"
     con un embedding agregado por asignatura.
  2. hybrid_search(query_text, top_k=5)
       → semantic top-10 sobre 321 docs
       + BM25 top-10 sobre 321 docs
       → RRF → top_k asignaturas completas
  3. process_syllabus(syllabus_text, top_courses=5)
       → extrae CONTENTS + LEARNING_OUTCOMES del texto de entrada
       → UN hybrid_search con el texto agregado
       → retorna top 5 asignaturas WU

Dependencias: chromadb, rank-bm25, sentence-transformers
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path tanto al ejecutar como
# script directo (python pipeline/retrieval.py) como módulo.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from pipeline.chunker import _clean, _chunk_contents, _chunk_learning_outcomes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNKS_PATH  = Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks.json"
CHROMA_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed" / "chroma_db"
TEST_OUTPUT  = Path(__file__).resolve().parents[1] / "data" / "processed" / "retrieval_test.json"

MODEL_NAME       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME  = "wu_syllabi_courses"   # 321 docs, uno por asignatura
RELEVANT_SECTIONS = {"contents", "learning_outcomes"}

SEMANTIC_TOP_N = 10
BM25_TOP_N     = 10
RRF_K          = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Prefijo de chunk → regex para limpieza
_PREFIX_RE = re.compile(r"^\[[A-Z_]+\]\s+\S+\s+\|\s*")


# ---------------------------------------------------------------------------
# Tokenizador
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-záéíóúüñà-ÿ\w]+", re.IGNORECASE | re.UNICODE)

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1]


# ---------------------------------------------------------------------------
# Construcción del corpus agregado
# ---------------------------------------------------------------------------

def _build_corpus(chunks_path: Path) -> tuple[list[str], list[str], list[dict]]:
    """
    Lee chunks.json y construye UN documento por asignatura concatenando
    solo las secciones CONTENTS y LEARNING_OUTCOMES (sin prefijos).

    Devuelve tres listas paralelas: ids, texts, metadatas.
    """
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    ids, texts, metas = [], [], []

    for course in data:
        code    = course.get("code", "")
        name    = course.get("name", "")
        meta_c  = course.get("metadata", {})
        credits = meta_c.get("credits", "")
        ctype   = meta_c.get("type", "")
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
            continue  # saltar asignaturas sin contenido relevante

        aggregated_text = "\n\n".join(relevant_parts)
        ids.append(code)
        texts.append(aggregated_text)
        metas.append({
            "code":       code,
            "name":       name,
            "credits":    credits,
            "type":       ctype,
            "schedule":   json.dumps(schedule, ensure_ascii=False),
            "all_chunks": json.dumps(all_chunk_ids, ensure_ascii=False),
        })

    return ids, texts, metas


# ---------------------------------------------------------------------------
# Clase principal: HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Un embedding por asignatura WU. Búsqueda híbrida semántica + BM25 + RRF.
    """

    def __init__(self) -> None:
        log.info("Cargando modelo de embeddings (%s) …", MODEL_NAME)
        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        log.info("Construyendo corpus agregado desde %s …", CHUNKS_PATH)
        self._ids, self._texts, self._metas = _build_corpus(CHUNKS_PATH)
        log.info("Corpus: %d asignaturas con contenido relevante.", len(self._ids))

        # Colección Chroma: crear o reusar
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            self.collection = client.get_collection(COLLECTION_NAME)
            if self.collection.count() == len(self._ids):
                log.info("Colección '%s' ya existe con %d docs. Reusando.",
                         COLLECTION_NAME, self.collection.count())
            else:
                log.info("Colección '%s' existe pero incompleta (%d/%d). Reconstruyendo.",
                         COLLECTION_NAME, self.collection.count(), len(self._ids))
                client.delete_collection(COLLECTION_NAME)
                self.collection = client.create_collection(
                    COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
                )
                self._index_collection()
        else:
            log.info("Creando colección '%s' con %d documentos …",
                     COLLECTION_NAME, len(self._ids))
            self.collection = client.create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            self._index_collection()

        # Índice BM25 sobre los mismos 321 documentos
        log.info("Construyendo índice BM25 …")
        self.bm25 = BM25Okapi([_tokenize(t) for t in self._texts])

        log.info("HybridRetriever listo (%d asignaturas).", len(self._ids))

    def _index_collection(self) -> None:
        """Genera embeddings y hace upsert en Chroma en batches de 32."""
        BATCH = 32
        total = len(self._ids)
        for start in range(0, total, BATCH):
            end    = min(start + BATCH, total)
            b_ids  = self._ids[start:end]
            b_txt  = self._texts[start:end]
            b_meta = self._metas[start:end]
            b_emb  = self.model.encode(b_txt, batch_size=BATCH,
                                       convert_to_numpy=True).tolist()
            self.collection.upsert(
                ids=b_ids, documents=b_txt, embeddings=b_emb, metadatas=b_meta
            )
            log.info("  Indexados %d/%d …", end, total)
        log.info("Indexación completa: %d documentos en Chroma.", self.collection.count())

    # ------------------------------------------------------------------
    # Búsqueda semántica
    # ------------------------------------------------------------------

    def _semantic_search(self, query_text: str, n: int) -> list[tuple[str, float]]:
        query_emb = self.model.encode(query_text, convert_to_numpy=True).tolist()
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=min(n, self.collection.count()),
            include=["distances"],
        )
        return [
            (cid, round(1.0 - dist, 6))
            for cid, dist in zip(res["ids"][0], res["distances"][0])
        ]

    # ------------------------------------------------------------------
    # Búsqueda BM25
    # ------------------------------------------------------------------

    def _bm25_search(self, query_text: str, n: int) -> list[tuple[str, float]]:
        tokens = _tokenize(query_text)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        top_sc  = [scores[i] for i in top_idx]
        max_sc  = max(top_sc) if top_sc and max(top_sc) > 0 else 1.0
        return [
            (self._ids[i], round(top_sc[r] / max_sc, 6))
            for r, i in enumerate(top_idx)
        ]

    # ------------------------------------------------------------------
    # RRF
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf(
        semantic: list[tuple[str, float]],
        bm25: list[tuple[str, float]],
    ) -> dict[str, dict]:
        sem_rank  = {cid: r + 1 for r, (cid, _) in enumerate(semantic)}
        sem_score = dict(semantic)
        bm25_rank  = {cid: r + 1 for r, (cid, _) in enumerate(bm25)}
        bm25_score = dict(bm25)

        fusion: dict[str, dict] = {}
        for cid in set(sem_rank) | set(bm25_rank):
            sr = sem_rank.get(cid,  SEMANTIC_TOP_N + 1)
            br = bm25_rank.get(cid, BM25_TOP_N + 1)
            fusion[cid] = {
                "semantic_score": round(sem_score.get(cid,  0.0), 4),
                "bm25_score":     round(bm25_score.get(cid, 0.0), 4),
                "rrf_score":      round(1 / (RRF_K + sr) + 1 / (RRF_K + br), 6),
            }
        return fusion

    # ------------------------------------------------------------------
    # hybrid_search — API pública
    # ------------------------------------------------------------------

    def hybrid_search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Búsqueda híbrida sobre 321 documentos agregados.
        Retorna top_k asignaturas WU con todos sus campos.
        """
        sem   = self._semantic_search(query_text, SEMANTIC_TOP_N)
        bm25  = self._bm25_search(query_text, BM25_TOP_N)
        fusion = self._rrf(sem, bm25)

        # Filtro de calidad: descartar resultados sin señal semántica real
        sorted_ids = [
            cid for cid in sorted(fusion, key=lambda c: fusion[c]["rrf_score"], reverse=True)
            if fusion[cid]["semantic_score"] >= 0.4
        ]

        results: list[dict] = []
        for rank, cid in enumerate(sorted_ids[:top_k], 1):
            idx  = self._ids.index(cid)
            meta = self._metas[idx]
            results.append({
                "rank":     rank,
                "code":     meta["code"],
                "name":     meta["name"],
                "credits":  meta["credits"],
                "type":     meta["type"],
                "section":  "contents+learning_outcomes",
                "text":     self._texts[idx],
                "schedule": json.loads(meta.get("schedule", "[]")),
            })
        return results

    # ------------------------------------------------------------------
    # process_query_course — API pública
    # ------------------------------------------------------------------

    def process_query_course(self, course: dict, top_courses: int = 5) -> list[dict]:
        """
        Recibe un course dict con al menos 'contents' y opcionalmente
        'learning_outcomes' y 'source'.

        Si source == "uam"  → solo contents para el embedding/BM25.
        Si source == "wu"   → contents + learning_outcomes.
        Sin source          → contents + learning_outcomes (comportamiento por defecto).
        """
        source   = course.get("source", "wu")
        contents = _clean(course.get("contents", ""))
        lo       = _clean(course.get("learning_outcomes", ""))

        if source == "uam":
            query_text = contents
            log.info("Fuente UAM: usando solo contents (%d chars).", len(query_text))
        else:
            query_text = "\n\n".join(filter(None, [contents, lo]))
            log.info("Fuente WU: usando contents + learning_outcomes (%d chars).", len(query_text))

        if not query_text:
            raise ValueError("El curso no tiene contenido (contents vacío).")

        return self.hybrid_search(query_text, top_k=top_courses)

    # Alias para compatibilidad con código anterior
    def process_syllabus(self, syllabus_text: str, top_courses: int = 5) -> list[dict]:
        return self.process_query_course(
            {"contents": syllabus_text, "source": "uam"},
            top_courses=top_courses,
        )


# ---------------------------------------------------------------------------
# Script de prueba
# ---------------------------------------------------------------------------

def _run_test(retriever: HybridRetriever) -> None:
    """
    Carga la asignatura UAM 16789 (MODELOS DE SIMULACIÓN EMPRESARIAL)
    desde data/my_courses/16789.json y ejecuta process_query_course().
    Si no existe, usa el primer curso WU del chunks.json como fallback.
    """
    my_course_path = _PROJECT_ROOT / "data" / "my_courses" / "16789.json"

    if my_course_path.exists():
        course = json.loads(my_course_path.read_text(encoding="utf-8"))
        log.info("Cargado curso UAM: [%s] %s", course.get("code"), course.get("name"))
    else:
        log.warning("%s no encontrado — usando primer curso WU como fallback.", my_course_path)
        chunks_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        first = chunks_data[0]
        parts = [
            _PREFIX_RE.sub("", c["text"]).strip()
            for c in first["chunks"] if c["section"] in RELEVANT_SECTIONS
        ]
        course = {
            "code":     first["code"],
            "name":     first["name"],
            "credits":  first.get("metadata", {}).get("credits", ""),
            "contents": "\n\n".join(parts),
            "source":   "wu",
        }

    t0      = time.time()
    results = retriever.process_query_course(course, top_courses=5)
    elapsed = time.time() - t0

    # ---------- Print ----------
    source_label = "UAM" if course.get("source") == "uam" else "WU"
    print(f"\n{'='*64}")
    print(f"  Query [{source_label}]: [{course.get('code')}] {course.get('name')}")
    print(f"  Tiempo: {elapsed:.2f}s")
    print(f"  {'─'*60}")
    print(f"  Top 5 asignaturas WU equivalentes")
    print(f"  {'─'*60}")

    for r in results:
        snippet = r["text"][:130].replace("\n", " ")
        print(f"\n  [{r['rank']}] {r['code']} — {r['name']}")
        print(f"       ECTS: {r['credits']}  |  Tipo: {r['type']}")
        print(f"       \"{snippet}…\"")

    print(f"\n{'='*64}")

    # ---------- Guardar ----------
    TEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUTPUT.write_text(
        json.dumps({
            "query_course":    {"code": course.get("code"), "name": course.get("name"),
                                "source": course.get("source", "uam")},
            "elapsed_seconds": round(elapsed, 3),
            "top_matches":     results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Resultados guardados en %s", TEST_OUTPUT)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = HybridRetriever()
    _run_test(retriever)
