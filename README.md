# 🎓 WU Matcher

**RAG-based Course Matching Tool for Erasmus Learning Agreements**

Una herramienta full-stack de Inteligencia Artificial diseñada para automatizar el doloroso proceso burocrático de los Erasmus Learning Agreements. 

Compara automáticamente los planes de estudio de la universidad de origen (UAM Madrid) con los de destino (WU Vienna) superando barreras de idioma y formato, para finalmente autogenerar el documento oficial en Excel y poder irte de Erasmus sin dolores de cabeza para convalidar asignaturas obligatorias o especializarte y escoger las optativas que más se asemejen a tu perfil.

---

## 🎯 El Problema vs. La Solución

**El Problema:** Los estudiantes de intercambio deben comparar manualmente cientos de guías docentes en diferentes idiomas y formatos, buscar solapamientos de temario y rellenar plantillas Excel estrictas.
**La Solución (WU Matcher):** Un sistema *end-to-end* que raspa los catálogos, vectoriza los contenidos, utiliza RAG Híbrido para encontrar las mejores equivalencias y usa LLMs para justificar las convalidaciones y generar el papeleo automáticamente.

---

## 🏗️ Arquitectura del Sistema

El proyecto se divide en módulos independientes para garantizar la escalabilidad y el mantenimiento:

### 1. Capa de Datos (Offline Processing)
- **Scraping (Selenium + BS4):** Extracción automatizada de 321 asignaturas del catálogo de la WU y sus respectivos *syllabi*.
- **Almacenamiento Local:** Los datos crudos y procesados residen en `data/` estructurados en JSON.
- **Base de Datos Vectorial:** Implementación de **ChromaDB** utilizando el espacio de similitud de coseno (HNSW).
- **Modelo de Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` para mapear conceptos en español e inglés al mismo espacio latente. Tiene que ser bilingüe ya que en la UAM las guías docentes están en español y en Viena en inglés/alemán.

### 2. Pipeline de Retrieval (`pipeline/`)
- **Chunking Estructural:** Troceo inteligente de documentos. Solo las secciones de `contents` y `learning_outcomes` se vectorizan para evitar el ruido generado por metodologías de evaluación o asistencia. (Gracias a la buena estructuración de la página web de la universidad de Viena).
- **Hybrid Search:** Búsqueda combinada que une precisión semántica y palabras clave.
  - *Búsqueda Semántica:* Similitud de coseno contra ChromaDB.
  - *Búsqueda Léxica:* `BM25Okapi` sobre los documentos tokenizados.
- **Fusión y Filtrado (RRF):** Implementación de *Reciprocal Rank Fusion* (K=60) combinando ambos rankings. Se aplica un *threshold* estricto de score ≥ 0.4 para eliminar falsos positivos.

---

## 🔀 Flujos de Ejecución (Backend FastAPI)

La API cuenta con 9 endpoints que orquestan dos lógicas de negocio completamente distintas:

### Flujo A: Asignaturas Obligatorias (Strict RAG)
Diseñado para encontrar coincidencias exactas de temario.
1. **Upload:** Recepción de PDFs de la UAM.
2. **Parsing:** Extracción limpia de la sección "1.13 Contenidos" mediante Regex y fallback de LLM (soportando multi-página).
3. **Retrieval:** Búsqueda Híbrida (Vectores + BM25) para obtener el Top 5 de la WU.
4. **Justificación LLM:** Prompt inyectado con los contenidos de ambas universidades. El modelo devuelve un JSON estructurado con el % de *match*, recomendaciones (SÍ/REVISAR/NO) y análisis de *gaps*.

### Flujo B: Asignaturas Optativas (Interest-Based Ranking)
Diseñado para priorizar los intereses profesionales del alumno.
1. **Clasificación Multi-label:** Un LLM clasifica la asignatura de origen en una o varias de las 12 áreas temáticas predefinidas (Ej: IA, FINANZAS, MARKETING).
2. **Filtrado Duro:** Se descartan las asignaturas de la WU que no compartan al menos un área temática.
3. **Ranking Semántico:** Se vectorizan los intereses explícitos del usuario y se calcula la similitud de coseno contra el catálogo filtrado para obtener un Top 10 de afinidad.
4. **Reporte Markdown:** Generación de pros, contras y solapamientos de las selecciones finales.

### Flujo C: Exportación y Utilidades
- **Búsqueda Manual:** Búsqueda difusa y exacta por código para casos límite.
- **Generación de Excel:** Creación asíncrona del *Learning Agreement* (plantilla UAM) inyectando datos con `openpyxl`. Implementa un retraso secuencial de 2s para evitar errores HTTP 429 de límite de peticiones (Rate Limit).

---

## 💻 Frontend (Next.js)

Interfaz de usuario fluida con rutas secuenciales (`/`, `/selector`, `/match`, `/optativas`):
- **Onboarding Interactivo:** Selección de áreas e intereses.
- **Visualización de Resultados:** Tarjetas interactivas con reportes de LLM y opciones de selección manual (con *debounce* de 300ms).
- **Detección de Conflictos:** Modal de calendario interactivo (Oct 2025 - Ene 2026) que renderiza *slots* de horarios detectando visualmente solapamientos entre las asignaturas seleccionadas.

---

## 🛠️ Stack Tecnológico

| Capa | Tecnologías |
| :--- | :--- |
| **Frontend** | Next.js 15, TypeScript, CSS-in-JS |
| **Backend** | FastAPI, Python 3.12 |
| **Base de Datos** | ChromaDB (Vectorial), JSON Local (Relacional) |
| **Modelos IA** | Ollama Cloud, `paraphrase-multilingual-MiniLM-L12-v2`, `gpt-oss:120b` |
| **Data Engineering** | Selenium, BeautifulSoup4, pdfplumber |

---

## 📊 Estado del Sistema

| Componente | Estado |
| :--- | :--- |
| Scraping WU (321 cursos) | ✅ Completado (Offline) |
| Embeddings (ChromaDB) | ✅ Indexados |
| Clasificación Optativas | ✅ Pre-clasificadas |
| Flujo Obligatorias | ✅ Funcional |
| Flujo Optativas | ✅ Funcional |
| Búsqueda Manual | ✅ Funcional |
| Generación de Excel | ✅ Funcional |
| Almacenamiento Persistente | ⚠️ JSON local (Pendiente migración) |
| Autenticación de Usuarios | ❌ No implementado (Uso personal) |

---

## 🚀 Roadmap Futuro

- Migración de almacenamiento JSON local a PostgreSQL + Supabase (pgvector).
- Implementación de un sistema de colas (Celery + Redis) para el procesamiento en lote.
- Despliegue de Autenticación de usuarios para abrir la herramienta a otros estudiantes.

---
*Made with ❤️ by Pablo*
