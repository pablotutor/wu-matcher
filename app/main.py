"""
main.py
-------
Punto de entrada principal de wu-matcher: interfaz de línea de comandos (CLI)
para realizar consultas sobre equivalencias de asignaturas UAM ↔ universidad destino.

Responsabilidades:
- Inicializar los módulos de retrieval y generación al arrancar.
- Leer las guías docentes propias del usuario desde data/my_courses/.
- Presentar un bucle interactivo donde el usuario introduce consultas en lenguaje natural.
- Orquestar el pipeline completo:
    1. Recuperar chunks relevantes (retrieval.py).
    2. Generar respuesta con contexto (generator.py).
    3. Mostrar respuesta y fuentes al usuario.
- Soportar modos: interactivo (REPL), consulta única (--query "...") y batch.

Dependencias: pipeline/retrieval.py, rag/generator.py, argparse (stdlib)
"""
