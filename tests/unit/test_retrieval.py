import sys
import os
import uuid

import pytest
from qdrant_client.models import VectorParams, Distance, PointStruct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.db.qdrant_setup import client
from src.pipeline.embedding import embed, EMBEDDING_SIZE

TEST_COLLECTION = "edge_rag_test_retrieval"

FIXTURE_CHUNKS = [
    {"text": "Internship agreement for college students", "source": "doc.pdf", "page_num": 1, "chunk_idx": 0},
    {"text": "Borscht recipe: beets, potatoes, cabbage, meat", "source": "doc.pdf", "page_num": 1, "chunk_idx": 1},
    {"text": "Instructions for installing Python and pip on Windows", "source": "doc.pdf", "page_num": 2, "chunk_idx": 2},
    {"text": "Rules for students completing academic and work internships", "source": "doc.pdf", "page_num": 2, "chunk_idx": 3},
    {"text": "History of space exploration in the 20th century", "source": "doc.pdf", "page_num": 3, "chunk_idx": 4},
]


@pytest.fixture(scope="module", autouse=True)
def setup_test_collection():
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)

    client.create_collection(
        collection_name=TEST_COLLECTION,
        vectors_config={"dense": VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE)},
    )

    vectors = embed([c["text"] for c in FIXTURE_CHUNKS])
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": vectors[i]},
            payload=FIXTURE_CHUNKS[i],
        )
        for i in range(len(FIXTURE_CHUNKS))
    ]
    client.upsert(collection_name=TEST_COLLECTION, points=points)

    yield

    client.delete_collection(TEST_COLLECTION)


def search_in_test_collection(question: str, top_k: int = 5) -> list[dict]:
    query_vector = embed(question)
    results = client.query_points(
        collection_name=TEST_COLLECTION,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points
    return [{"text": p.payload["text"], "score": p.score} for p in results]


class TestSearch:

    def test_finds_relevant_chunk_about_internship(self):
        results = search_in_test_collection("student internship agreement", top_k=3)
        texts = [r["text"] for r in results]
        assert any("internship" in t.lower() for t in texts)

    def test_finds_relevant_chunk_about_python(self):
        results = search_in_test_collection("how to install Python", top_k=3)
        texts = [r["text"] for r in results]
        assert any("Python" in t for t in texts)

    def test_top_k_limits_number_of_results(self):
        results = search_in_test_collection("any text", top_k=2)
        assert len(results) == 2

    def test_results_sorted_by_descending_score(self):
        results = search_in_test_collection("student internship", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])