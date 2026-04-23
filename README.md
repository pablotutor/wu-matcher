# WU Matcher
> RAG-based Course Matching Tool for Erasmus Learning Agreements

A full-stack AI tool that automates the most painful part of Erasmus paperwork: finding which WU Vienna courses are equivalent to your home university (UAM Madrid) courses, and generating the Learning Agreement Excel automatically.

Built for my own exchange semester — then turned into a portfolio project to document the AI engineering decisions behind it.

## Overview

When preparing for an Erasmus exchange, students must manually compare syllabi across two universities, in different languages, with no standardised format — then fill out a bureaucratic Excel document. This tool automates that process end-to-end.

The system handles two distinct matching problems with different architectures:

- **Obligatorias (required courses):** strict RAG matching — embeds UAM syllabi, retrieves semantically similar WU courses, then an LLM analyses topic overlap and outputs a structured recommendation (SÍ / REVISAR / NO).
- **Optativas (electives):** interest-based ranking — user declares professional interests, LLM classifies both UAM and WU courses into 12 thematic areas, then cosine similarity ranks the top 10 WU matches per elective.

## Features

- Batch PDF upload of UAM course guides (guías docentes)
- Automatic extraction of section 1.13 (Contenidos) via regex + LLM fallback, handling multi-page syllabi
- Hybrid retrieval: semantic embeddings + BM25 with RRF fusion
- Two matching pipelines with different logic for required vs elective courses
- Multi-label classification into 12 thematic areas (a course can be IA + ENTREPRENEURSHIP simultaneously)
- Interactive calendar (Oct 2025 – Jan 2026) for detecting WU schedule conflicts
- LLM report with pros/cons analysis per selected course pairing
- Automatic Learning Agreement generation in Excel (official UAM template format)
- Manual search fallback: if the Top 10 doesn't have what you need, search any WU course by code or name

## Tech Stack

| Layer | Stack |
|-------|-------|
| **Frontend** | Next.js 15 · TypeScript · CSS-in-JS |
| **Backend** | FastAPI · Python 3.12 |
| **Vector DB** | ChromaDB |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` |
| **LLM** | `gpt-oss:120b` via Ollama Cloud |
| **Scraping** | Selenium · BeautifulSoup4 · pdfplumber |
| **Data** | JSON (local) · planned migration to PostgreSQL + Supabase |

## Project Structure

```
wu-matcher/
├── scraper/
│   ├── phase1_catalog.py       # Selenium: WU course catalog + links
│   ├── phase2_syllabi.py       # BS4: syllabus content extraction
│   └── parse_my_syllabus.py    # UAM PDF parser (regex + LLM)
├── pipeline/
│   ├── chunker.py              # Structural chunking by section
│   ├── embeddings.py           # Embed + index into ChromaDB
│   └── retrieval.py            # Hybrid semantic + BM25 retrieval
├── rag/
│   └── generator.py            # LLM generation layer
├── app/
│   ├── api.py                  # FastAPI — all endpoints
│   └── frontend/               # Next.js web app
├── data/
│   ├── raw/                    # 321 WU courses (JSON)
│   ├── my_courses/             # UAM syllabi (PDF + parsed JSON)
│   └── processed/              # ChromaDB embeddings + BM25 index
└── scripts/                    # Utility and migration scripts
```

## Setup

### Requirements

- Python 3.10+
- Node.js 18+
- Ollama Cloud API key

### Backend

```bash
cd wu-matcher
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
echo "OLLAMA_API_KEY=your_key" > .env

# Start API
uvicorn app.api:app --reload --port 8000
```

### Frontend

```bash
cd app/frontend
npm install
npm run dev
# Open http://localhost:3000
```

## Pipeline Walkthrough

### Obligatorias — Strict RAG

```
Upload PDF → Extract section 1.13 (regex, multi-page aware)
         → Embed with MiniLM-L12-v2
         → Hybrid retrieval: semantic + BM25 → top 5 WU candidates
         → LLM analysis: overlapping topics, gaps, match %
         → Output: SÍ convalidar / REVISAR / NO
```

The chunking strategy matters here: including assessment/attendance sections introduced noise. Only `contents` + `learning_outcomes` are embedded.

Semantic score filtering: RRF fusion requires a ≥ 0.4 threshold to eliminate false positives from unrelated courses with shared vocabulary.

### Optativas — Interest-Based Ranking

```
User selects interests: ["IA", "DATA_SCIENCE", ...]
         → Upload UAM elective PDFs
         → LLM multi-label classification (12 fixed categories)
         → Filter: WU courses in matching thematic areas
         → Rank: cosine similarity (user interests embedding vs WU embedding)
         → Top 10 per elective → user selects manually
         → LLM report: pro/con analysis per pairing
         → Excel Learning Agreement auto-generated
```

Multi-label classification was necessary because courses like "AI for Business" belong to both IA and ENTREPRENEURSHIP simultaneously. Single-label classification lost too many valid matches.

## Key Engineering Decisions

**Why not just cosine similarity for required courses?**
Required courses need a strict content match, not a thematic match. A WU course on "Data Structures" and a UAM course on "Algorithms" are semantically close but may not satisfy accreditation requirements. The LLM analysis layer adds the structured reasoning that pure vector search can't provide.

**Why Ollama Cloud instead of OpenAI?**
`gpt-oss:120b` gives GPT-4 class performance at a fraction of the cost for this use case, with no data retention policy concerns for academic documents.

**Why ChromaDB over Pinecone/Weaviate?**
Local-first development without infrastructure overhead. The dataset (321 courses) doesn't justify managed vector DB costs. Migration path to Supabase pgvector is planned.

**Rate limiting on Learning Agreement generation**
The Excel generation calls the LLM once per course pair. Sequential processing with 2s delays was implemented after hitting 429 errors with parallel calls (10 courses × 2 calls = 20 simultaneous requests).

## Roadmap

- [ ] PostgreSQL + Supabase (replace local JSON storage)
- [ ] Batch processing with job queue (Celery + Redis)
- [ ] Multi-university support (KU Leuven, Bocconi, etc.)
- [ ] Fine-tune embeddings on academic domain data
- [ ] Collaborative filtering — improve recommendations from user selections
- [ ] Mobile app (React Native)

## License

MIT

## Author

Pablo López · [Building in public](https://github.com/pablotutor)
