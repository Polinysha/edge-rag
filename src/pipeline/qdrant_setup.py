"""
Подключение к Qdrant и создание коллекции для гибридного поиска
(dense-вектор для смыслового поиска + sparse-вектор для BM25-поиска по словам).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
)

COLLECTION_NAME = "edge_rag"
DENSE_VECTOR_SIZE = 384  # размерность для sentence-transformers/all-MiniLM-L6-v2

client = QdrantClient(url="http://localhost:6333")


def create_collection():
    """Создаёт коллекцию с dense и sparse полями, если она ещё не существует."""
    if client.collection_exists(COLLECTION_NAME):
        print(f"Коллекция '{COLLECTION_NAME}' уже существует, пересоздание не требуется.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_VECTOR_SIZE, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )
    print(f"Коллекция '{COLLECTION_NAME}' создана.")


if __name__ == "__main__":
    create_collection()
    info = client.get_collection(COLLECTION_NAME)
    print(info)
