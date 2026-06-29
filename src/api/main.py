"""
FastAPI service exposing the Edge-RAG pipeline over HTTP.
"""

import os
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.pipeline.extraction import extract_pdf
from src.pipeline.chunking import chunk_pages
from src.pipeline.indexing import index_chunks
from src.pipeline.pipeline import ask as ask_pipeline

app = FastAPI(title="Edge-RAG API")

VERSION = "0.1.0"
DATA_DIR = "data"


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    """Simple liveness check — doesn't touch the LLM or Qdrant."""
    return {"status": "ok", "version": VERSION}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accepts a PDF, saves it to data/, and runs it through the indexing
    pipeline synchronously: extract -> chunk -> embed -> upsert into Qdrant.
    """
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
def ask(request: AskRequest):
    """
    Answers a question against whatever is currently indexed in Qdrant.
    Right now this calls the simple ask() pipeline from Task 3 (search + generate,
    no LangGraph yet). When the graph lands in Task 5, only this implementation
    changes — the request/response contract stays the same.
    """
    result = ask_pipeline(request.question)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metrics": {},  # populated once RAGAS evaluation is wired in
    }