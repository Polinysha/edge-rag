from src.db.qdrant_setup import client, COLLECTION_NAME
from src.pipeline.embedding import embed


def search(question: str, top_k: int = 5) -> list[dict]:

    query_vector = embed(question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points

    return [
        {
            "text": point.payload["text"],
            "source": point.payload["source"],
            "page_num": point.payload["page_num"],
            "chunk_idx": point.payload["chunk_idx"],
            "score": point.score,
        }
        for point in results
    ]


if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "test question"

    results = search(question, top_k=5)
    print(f"Question: {question!r}")
    print(f"Results found: {len(results)}")
    print()
    for r in results:
        print(f"score={r['score']:.4f}  page={r['page_num']}  chunk_idx={r['chunk_idx']}")
        print(f"  {r['text'][:150]}")
        print()