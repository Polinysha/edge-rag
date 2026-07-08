from rank_bm25 import BM25Okapi

from src.db.qdrant_setup import client, COLLECTION_NAME
from src.pipeline.state import State
from src.pipeline.parent_store import get_parent_blocks

DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FINAL_TOP_K = 5
RRF_K = 60

# Flag to enable Parent-Child retrieval (set via config or environment)
USE_PARENT_CHILD = False


def dense_search(query_vector: list[float], top_k: int = DENSE_TOP_K) -> list[dict]:
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points
    return [point.payload for point in results]


def sparse_search(question: str, top_k: int = SPARSE_TOP_K) -> list[dict]:

    all_points = client.scroll(collection_name=COLLECTION_NAME, limit=10_000)[0]
    payloads = [point.payload for point in all_points]

    if not payloads:
        return []

    tokenized_corpus = [p["text"].lower().split() for p in payloads]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [payloads[i] for i in ranked_indices[:top_k]]


def chunk_key(chunk: dict) -> tuple:
    return (chunk["source"], chunk["page_num"], chunk["chunk_idx"])


def rrf_merge(dense_results: list[dict], sparse_results: list[dict], k: int = RRF_K) -> list[dict]:

    scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}

    for rank, chunk in enumerate(dense_results, start=1):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks_by_key[key] = chunk

    for rank, chunk in enumerate(sparse_results, start=1):
        key = chunk_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        chunks_by_key[key] = chunk

    ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
    return [chunks_by_key[key] for key in ranked_keys]


def retrieve(state: State) -> dict:

    dense_results = dense_search(state["query_vector"])
    sparse_results = sparse_search(state["question"])

    merged = rrf_merge(dense_results, sparse_results)
    top_results = merged[:FINAL_TOP_K]
    
    # Parent-Child mode: return parent blocks instead of child chunks
    if USE_PARENT_CHILD and top_results and "parent_id" in top_results[0]:
        parent_ids = [chunk["parent_id"] for chunk in top_results if "parent_id" in chunk]
        parent_blocks = get_parent_blocks(parent_ids)
        
        if parent_blocks:
            return {"documents": parent_blocks}
    
    return {"documents": top_results}


if __name__ == "__main__":
    from src.pipeline.embedding import embed

    question = "what is this document about"
    state = {"question": question, "query_vector": embed(question)}
    result = retrieve(state)

    print(f"Question: {question}")
    print(f"Top {len(result['documents'])} merged results:")
    for doc in result["documents"]:
        print(f"  page={doc['page_num']}  {doc['text'][:100]}")
