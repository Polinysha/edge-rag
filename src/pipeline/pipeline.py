"""
Simple end-to-end pipeline (no LangGraph yet): question -> search -> generate -> answer.
Wrapped with @traceable so the whole call shows up as a single trace in LangSmith,
with search and generate visible as nested steps.
"""

from langsmith import traceable

from src.pipeline.retrieval import search
from src.pipeline.generation import generate


@traceable
def ask(question: str, top_k: int = 5) -> dict:
    """
    Runs the full pipeline for a single question.
    Returns {"answer": str, "sources": [{"source": str, "page_num": int}, ...]}.
    """
    chunks = search(question, top_k=top_k)
    answer = generate(question, chunks)

    sources = [
        {"source": c["source"], "page_num": c["page_num"]}
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "what is this document about"

    result = ask(question)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
