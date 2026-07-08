# Edge-RAG

A production-ready RAG pipeline over PDFs with hybrid search, query expansion, and an evaluation loop built on RAGAS. The core idea is a LangGraph-based system that can correct itself: if retrieval returns irrelevant chunks, the graph rewrites the query and searches again instead of generating a hallucinated answer from a bad context.

## Architecture & Layout

The project follows the Separation of Concerns principle, keeping the web layer, business logic, and data layer strictly isolated.

```text
src/
  api/           FastAPI application (HTTP contracts and routing only)
  ui/            Streamlit frontend (communicates via HTTP API)
  pipeline/      LangGraph core logic (state, nodes, and conditional edges)
  chunking/      Text splitters and advanced chunking (Parent-Child, Structural)
  db/            Qdrant setup and vector operations
  llm/           OpenRouter client and prompt management
tests/
  unit/          Isolated tests for functions and graph nodes
  api/           API contract tests (endpoints, status codes, JSON shapes)
  e2e/           End-to-end tests evaluating the RAG pipeline with RAGAS
data/            PDFs for development and testing (ignored in git)