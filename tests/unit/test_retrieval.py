"""
Юнит-тесты на retrieval.py (search).

Использует ОТДЕЛЬНУЮ тестовую коллекцию (не edge_rag), чтобы:
- тест был детерминированным (известно, что именно должно найтись)
- тест не зависел от того, что сейчас лежит в рабочей коллекции
- тест не портил рабочие данные

Запуск:
    uv run pytest tests/unit/test_retrieval.py -v

Требует запущенный Qdrant на localhost:6333.
"""

import sys
import os
import uuid

import pytest
from qdrant_client.models import VectorParams, Distance, PointStruct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.qdrant_setup import client
from src.pipeline.embedding import embed, EMBEDDING_SIZE

TEST_COLLECTION = "edge_rag_test_retrieval"

# Заранее известные чанки: на вопрос про "договор о практике" должен находиться chunk 0,
# на вопрос про "погоду" — ничего релевантного из этого набора.
FIXTURE_CHUNKS = [
    {"text": "Договор об организации практики учащихся в колледже", "source": "doc.pdf", "page_num": 1, "chunk_idx": 0},
    {"text": "Рецепт борща: свёкла, картофель, капуста, мясо", "source": "doc.pdf", "page_num": 1, "chunk_idx": 1},
    {"text": "Инструкция по установке Python и pip на Windows", "source": "doc.pdf", "page_num": 2, "chunk_idx": 2},
    {"text": "Правила прохождения учебной и производственной практики студентами", "source": "doc.pdf", "page_num": 2, "chunk_idx": 3},
    {"text": "История развития космонавтики в XX веке", "source": "doc.pdf", "page_num": 3, "chunk_idx": 4},
]


@pytest.fixture(scope="module", autouse=True)
def setup_test_collection():
    """Создаёт тестовую коллекцию с фикстурными чанками перед тестами и удаляет после."""
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
    """Локальная копия search(), но указывающая на тестовую коллекцию."""
    query_vector = embed(question)
    results = client.query_points(
        collection_name=TEST_COLLECTION,
        query=query_vector,
        using="dense",
        limit=top_k,
    ).points
    return [{"text": p.payload["text"], "score": p.score} for p in results]


class TestSearch:

    def test_находит_релевантный_чанк_про_практику(self):
        results = search_in_test_collection("договор о практике студента", top_k=3)
        texts = [r["text"] for r in results]
        assert any("практики" in t for t in texts)

    def test_находит_релевантный_чанк_про_python(self):
        results = search_in_test_collection("как установить Python", top_k=3)
        texts = [r["text"] for r in results]
        assert any("Python" in t for t in texts)

    def test_top_k_ограничивает_число_результатов(self):
        results = search_in_test_collection("любой текст", top_k=2)
        assert len(results) == 2

    def test_результаты_отсортированы_по_убыванию_score(self):
        results = search_in_test_collection("практика студентов", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
