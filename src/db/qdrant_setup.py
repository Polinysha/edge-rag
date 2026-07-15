from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
)

COLLECTION_NAME = "edge_rag"
DENSE_VECTOR_SIZE = 384

client = QdrantClient(host="127.0.0.1", grpc_port=6334, prefer_grpc=True)


def create_collection():
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