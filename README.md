# Edge-RAG

A RAG pipeline over PDFs with hybrid search, query expansion, and an evaluation loop built on RAGAS. The core idea is a LangGraph graph that can correct itself: if retrieval comes back with irrelevant chunks, the graph rewrites the query and searches again instead of just generating an answer from bad context.

The project is still in progress. Right now the PDF extraction layer and the basic retrieval pipeline are done and tested. The graph itself, hybrid search, the API, and the UI are being built next.

## How it works (so far)

A PDF goes through `extract_pdf`, which checks every page for a text layer. If there's one, the text is pulled directly through PyMuPDF. If not, the page is treated as a scan and run through Tesseract OCR instead. The extracted text gets cleaned up — line breaks inside hyphenated words get joined back together, repeated whitespace gets collapsed, and stray separator lines get stripped out.

The cleaned text is split into chunks with `RecursiveCharacterTextSplitter` at 400 characters with 80 characters of overlap. Each chunk keeps track of which document, page, and position it came from, since that information is what eventually lets the system point back to a source.

Chunks get embedded with `all-MiniLM-L6-v2` running locally on CPU, and the resulting 384-dimensional vectors are stored in Qdrant alongside the chunk metadata. Search right now is plain dense vector search — you give it a question, it embeds the question the same way, and Qdrant returns the closest chunks by cosine similarity. Qdrant is also set up with a sparse vector field already, which is there for when BM25-style keyword search gets added on top of this for proper hybrid retrieval.

Where this is headed: the LangGraph graph will wrap retrieval in a loop that checks chunk quality before generating anything, rewrites the question if the chunks look off-topic, and only then calls the LLM. RAGAS will score the output on context relevance, faithfulness, and answer relevance once that's in place.

## What's not built yet

There's no LLM call anywhere in the code yet — no OpenRouter integration, no generation step. There's no LangGraph graph, so there's no retry logic and no query expansion either, even though the data model already has room for them. Hybrid search exists at the schema level in Qdrant but the BM25 side of it isn't wired up. There's no FastAPI service and no Streamlit UI. None of this is hidden or faked — it's just not there yet.

## Setting it up

You need Python 3.11 or newer, uv, Docker, and Tesseract OCR installed locally (with the Russian language pack, since the OCR call is currently hardcoded to `rus+eng`).

```bash
git clone https://github.com/Polinysha/edge-rag.git
cd edge-rag
uv venv --python 3.11
uv sync
```

Then create a `.env` file in the root with these variables:

```
OPENROUTER_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=edge-rag
```

The OpenRouter and LangSmith keys aren't used by anything yet, but they're already expected by the project structure for when generation and tracing get added.

Qdrant runs in Docker, with a volume so the index survives restarts:

```bash
docker run -d -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant
```

If you're on Windows and OCR throws `TesseractNotFoundError`, open `src/pipeline/extraction.py` and set the path to `tesseract.exe` explicitly near the top of the file — pytesseract doesn't always find it on its own on Windows.

## Running it

Everything inside `src` needs to be run as a module, not as a script by file path — otherwise Python won't resolve the `src` package correctly.

```bash
uv run python -m src.pipeline.extraction data/your.pdf
uv run python -m src.pipeline.qdrant_setup
uv run python -m src.pipeline.indexing data/your.pdf
uv run python -m src.pipeline.retrieval "your question"
```

`indexing` runs the whole pipeline end to end — extraction, chunking, embedding, and upserting into Qdrant — on a single file. `retrieval` just queries whatever is already indexed.

## Tests

```bash
uv run pytest tests/unit/ -v
```

There are 23 unit tests right now, covering text extraction and cleanup, chunk boundaries and metadata, embedding dimensionality and determinism, and search correctness. The retrieval tests run against a separate, disposable Qdrant collection that gets created and torn down around the test run, so they don't depend on or interfere with whatever you've actually indexed in the main collection.

There's nothing in `tests/api` or `tests/e2e` yet — those will show up once the API and the graph exist.

## Layout

```
src/
  pipeline/      extraction, chunking, embedding, indexing, retrieval, graph nodes (later)
  api/           empty for now
  ui/            empty for now
tests/
  unit/          tests for the pipeline functions above
  api/           empty for now
  e2e/           empty for now
data/            PDFs used for development and testing, not committed
```

## A couple of decisions worth explaining

Retrieval and generation are kept conceptually separate even at this stage, before generation exists. The reasoning is that once the LLM is in the loop, a bad answer could mean either that the wrong chunks were retrieved or that the LLM did a poor job with the right chunks — and those are different problems with different fixes. Building the retrieval evaluation logic in from the start, rather than bolting it on later, should make that distinction easier to test for.

The retrieval unit tests use their own throwaway Qdrant collection instead of the real one. It would have been faster to just test against whatever is already indexed, but that makes the test fragile — it would break the moment someone reindexes something different, and it wouldn't tell you anything reliable about whether search itself still works correctly.