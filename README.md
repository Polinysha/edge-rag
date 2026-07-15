# Edge-RAG

Edge-RAG is a robust Retrieval-Augmented Generation (RAG) pipeline designed for parsing documents, executing advanced retrieval, and running a self-correcting evaluation loop. The core architecture utilizes LangGraph to implement a self-reflective workflow: if the initial retrieval yields irrelevant context, the graph rewrites the query and re-executes the search rather than hallucinating an answer from poor context.

## System Architecture

*   **Extraction Layer**: PDFs are processed via `PyMuPDF`. If a text layer is absent, the pipeline automatically falls back to `Tesseract OCR` (configured for both English and Russian). The extracted text undergoes rigorous normalization (handling hyphenations, collapsing whitespace, and stripping artifacts).
*   **Chunking & Embedding**: Cleaned text is segmented using `RecursiveCharacterTextSplitter` (400 characters, 80-character overlap) to preserve document metadata (source, page, position). Chunks are embedded using `all-MiniLM-L6-v2` (running locally via CPU).
*   **Vector Store**: High-dimensional vectors and metadata are stored in `Qdrant`, running locally in Docker. The system utilizes dense vector search via cosine similarity, with schema-level preparations for BM25 hybrid search integration.
*   **Graph Execution**: `LangGraph` orchestrates the retrieval and generation loop. It validates chunk quality, triggers query expansion/rewriting when necessary, and interfaces with the LLM (powered by Groq) for final generation.
*   **Evaluation**: An automated evaluation loop built on `RAGAS` measures the pipeline's performance across key metrics: Context Relevance, Faithfulness, Answer Relevance, and Context Precision.

## Prerequisites

Ensure the following dependencies are installed on your system:
*   Python 3.11 or higher
*   `uv` (fast Python package installer and resolver)
*   Docker and Docker Compose
*   Tesseract OCR (must include the Russian language pack). 
    *Note for Windows users: If you encounter a `TesseractNotFoundError`, explicitly set the path to `tesseract.exe` inside `src/pipeline/extraction.py`.*

## Installation & Setup

1.  **Clone the repository and initialize the environment:**
    ```bash
    git clone [https://github.com/Polinysha/edge-rag.git](https://github.com/Polinysha/edge-rag.git)
    cd edge-rag
    uv venv --python 3.11
    uv sync
    ```
    
2.  **Configure Environment Variables:**
    Create a `.env` file in the project root with the following keys:
    ```env
    GROQ_API_KEY=your_groq_api_key
    LANGCHAIN_API_KEY=your_langchain_api_key
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_PROJECT=edge-rag
    ```

3.  **Initialize the Infrastructure:**
    Start Qdrant (and associated services) using Docker Compose. The storage volume ensures the index persists across restarts, and necessary ports (including gRPC) are exposed.
    ```bash
    docker compose up -d
    ```
ы
## Running the Pipeline

All modules within the `src` directory must be executed as Python modules to ensure correct package resolution.

**Data Ingestion & Indexing:**
Run the end-to-end ingestion pipeline (extraction, chunking, embedding, and Qdrant upsertion) on a single file:
```bash
uv run python -m src.pipeline.indexing data/your.pdf