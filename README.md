# Edge-RAG

RAG-пайплайн поверх PDF с гибридным поиском (dense + BM25), query expansion и evaluation loop через RAGAS. Архитектура — LangGraph-граф с self-correction: если retrieval приносит нерелевантные чанки, граф переформулирует вопрос и повторяет поиск, прежде чем генерировать ответ.

> **Статус:** в разработке. Реализованы извлечение текста из PDF (включая OCR) и базовый retrieval pipeline (chunking → embedding → indexing → search). LangGraph-граф, hybrid search, FastAPI и UI — в следующих итерациях.

## Архитектура (план)

```
PDF → extract_pdf → chunk_pages → embed → Qdrant
                                              ↓
                                     expand_query → retrieve → grade_documents
                                                                      ↓
                                                          (retry или) generate → evaluate
```

Полное описание графа и обоснование conditional edges появится здесь после Task 5.

## Стек

| Слой | Инструмент | Статус |
|---|---|---|
| PDF parsing | PyMuPDF | ✅ |
| OCR | Tesseract 5 (rus+eng) | ✅ |
| Chunking | langchain-text-splitters | ✅ baseline |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | ✅ |
| Vector DB | Qdrant (Docker) | ✅ dense, sparse-поле создано, BM25 в работе |
| LLM | OpenRouter (Llama 3.1 8B free) | ⏳ |
| Orchestration | LangGraph | ⏳ |
| Tracing | LangSmith | ⏳ |
| Evaluation | RAGAS + DeepEval | ⏳ |
| API | FastAPI | ⏳ |
| UI | Streamlit | ⏳ |

## Установка

Требуется Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker, [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (с языковым пакетом `rus`).

```bash
git clone https://github.com/Polinysha/edge-rag.git
cd edge-rag

uv venv --python 3.11
uv sync
```

Создать `.env` в корне проекта:

```
OPENROUTER_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=edge-rag
```

Поднять Qdrant:

```bash
docker run -d -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant
```

**Windows:** если OCR падает с `TesseractNotFoundError`, раскомментируй и поправь путь к `tesseract.exe` в начале `src/pipeline/extraction.py`.

## Использование

Все модули внутри `src` запускаются через `-m`, не напрямую по пути — иначе Python не резолвит пакет `src`:

```bash
# извлечь текст из PDF (native + OCR)
uv run python -m src.pipeline.extraction data/your.pdf

# создать коллекцию в Qdrant
uv run python -m src.pipeline.qdrant_setup

# проиндексировать документ (extract → chunk → embed → upsert)
uv run python -m src.pipeline.indexing data/your.pdf

# найти релевантные чанки по вопросу
uv run python -m src.pipeline.retrieval "твой вопрос"
```

## Тесты

```bash
uv run pytest tests/unit/ -v
```

23 теста: extraction (OCR/native, очистка текста), chunking (границы размера, метаданные, сквозная нумерация), embedding (размерность, детерминированность), retrieval (поиск на изолированной тестовой коллекции — не трогает рабочие данные в `edge_rag`).

API-тесты (`tests/api/`) и e2e-тесты графа на golden-датасете (`tests/e2e/`) появятся вместе с соответствующими компонентами.

## Структура проекта

```
src/
  pipeline/      ноды графа, чанкинг, embedding, indexing, retrieval
  api/           FastAPI-сервис (в разработке)
  ui/            Streamlit-интерфейс (в разработке)
tests/
  unit/          юнит-тесты на отдельные функции/ноды
  api/           тесты на контракт API (в разработке)
  e2e/           тесты графа на golden-датасете (в разработке)
data/            PDF для разработки и тестов — не в git
```

## Дизайн-решения

- **Retrieval отделён от generation в оценке качества.** Если ответ плохой — сначала проверяется, нашлись ли релевантные чанки вообще. Чинить промпт, когда виноват retrieval, — трата времени.
- **Изолированные тестовые коллекции для retrieval-тестов.** Юнит-тесты на `search` работают на отдельной коллекции с заранее известными чанками, а не на рабочей `edge_rag` — тест детерминирован и не зависит от того, что сейчас проиндексировано.
- **Каждая нода графа — отдельная функция, не больше 50 строк.** Логика графа (edges) не смешивается с логикой нод. Изменение одной ноды — отдельный PR.
