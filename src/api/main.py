import os
from fastapi import FastAPI, UploadFile, File
from langsmith import traceable
from pydantic import BaseModel

from src.pipeline.extraction import extract_pdf
from src.chunking.chunking import chunk_pages
from src.pipeline.indexing import index_chunks

app = FastAPI(title="Edge-RAG API")

VERSION = "0.1.0"
DATA_DIR = "data"


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "version": VERSION}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    pages = extract_pdf(save_path)
    chunks = chunk_pages(pages)
    indexed_count = index_chunks(chunks)

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks_indexed": indexed_count,
    }


@app.post("/ask")
@traceable(name="ask")
def ask(request: AskRequest):
    from src.pipeline.graph import get_graph

    graph = get_graph()
    state = graph.invoke({"question": request.question, "retry_count": 0})

    return {
        "answer": state.get("generation", ""),
        "sources": [
            {"source": doc["source"], "page_num": doc["page_num"]}
            for doc in state.get("documents", [])
        ],
        "metrics": {
            "context_relevance": state.get("context_relevance"),
            "faithfulness": state.get("faithfulness"),
            "answer_relevance": state.get("answer_relevance"),
        },
    }