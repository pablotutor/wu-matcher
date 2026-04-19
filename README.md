# wu-matcher

Sistema RAG para identificar equivalencias entre asignaturas de la UAM y las de una universidad de destino (programa de intercambio / Erasmus).

## Arquitectura

```
wu-matcher/
├── data/
│   ├── raw/          # JSON con datos scrapeados de la web
│   ├── processed/    # Chunks procesados + base vectorial ChromaDB
│   └── my_courses/   # Guías docentes propias (PDF/JSON)
├── scraper/
│   ├── phase1_catalog.py   # Selenium: lista de asignaturas + links
│   └── phase2_syllabi.py   # Requests + BS4: contenido de cada syllabus
├── pipeline/
│   ├── chunker.py          # Chunking estructural por secciones
│   ├── embeddings.py       # Embeddings con paraphrase-multilingual-MiniLM-L12-v2
│   └── retrieval.py        # Búsqueda híbrida semántica + BM25
├── rag/
│   └── generator.py        # Generación con Gemma 4 E2B vía Ollama
└── app/
    └── main.py             # CLI de consultas
```

## Setup

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Ejecutar pipeline completo
python scraper/phase1_catalog.py   # 1. Obtener catálogo
python scraper/phase2_syllabi.py   # 2. Extraer syllabi
python pipeline/chunker.py         # 3. Chunking
python pipeline/embeddings.py      # 4. Generar embeddings

# Consultar
python app/main.py --query "Algoritmos y estructuras de datos"
# o modo interactivo
python app/main.py
```

## Requisitos

- Python >= 3.10
- Ollama corriendo localmente con el modelo `gemma` descargado (`ollama pull gemma3`)
- ChromeDriver compatible con tu versión de Chrome (para Selenium)
