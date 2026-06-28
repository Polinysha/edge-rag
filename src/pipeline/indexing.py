"""
Indexing chunks into Qdrant: vectorization + batched point upload.
"""

import uuid
from qdrant_client.models import PointStruct

from src.pipeline.qdrant_setup import client, COLLECTION_NAME
from src.pipeline.embedding import embed

BATCH_SIZE = 64


def index_chunks(chunks: list[dict]) -> int:
    """
    Takes a list of chunks [{text, source, page_num, chunk_idx}, ...].
    Vectorizes texts, builds points, and sends them to Qdrant in batches.
    Returns the number of indexed chunks.
    """
    if not chunks:
        return 0

    total_indexed = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": vectors[j]},
                payload={
                    "text": batch[j]["text"],
                    "source": batch[j]["source"],
                    "page_num": batch[j]["page_num"],
                    "chunk_idx": batch[j]["chunk_idx"],
                },
            )
            for j in range(len(batch))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total_indexed += len(points)

    return total_indexed


if __name__ == "__main__":
    import sys
    from src.pipeline.extraction import extract_pdf
    from src.pipeline.chunking import chunk_pages
    from src.pipeline.qdrant_setup import create_collection

    create_collection()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"
    pages = extract_pdf(test_path)
    chunks = chunk_pages(pages)

    count = index_chunks(chunks)
    print(f"Indexed chunks: {count}")

    total_in_collection = client.count(COLLECTION_NAME).count
    print(f"Total points in collection '{COLLECTION_NAME}': {total_in_collection}")