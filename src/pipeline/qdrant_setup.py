"""
Qdrant connection and collection creation for hybrid search
(dense vector for semantic search + sparse vector for BM25 keyword search).
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
)

COLLECTION_NAME = "edge_rag"
DENSE_VECTOR_SIZE = 384  # dimension for sentence-transformers/all-MiniLM-L6-v2

client = QdrantClient(url="http://localhost:6333")


def create_collection():
    """Creates the collection with dense and sparse fields if it doesn't exist yet."""
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping creation.")
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
    print(f"Collection '{COLLECTION_NAME}' created.")


if __name__ == "__main__":
    create_collection()
    info = client.get_collection(COLLECTION_NAME)
    print(info)