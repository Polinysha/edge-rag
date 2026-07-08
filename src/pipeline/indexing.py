import uuid
from qdrant_client.models import PointStruct

from src.db.qdrant_setup import client, COLLECTION_NAME
from src.pipeline.embedding import embed

BATCH_SIZE = 64


def index_chunks(chunks: list[dict]) -> int:

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


def index_parent_child_children(children: list[dict], parents: list[dict]) -> int:

    from src.pipeline.parent_store import store_parent_block
    
    if not children:
        return 0

    for parent in parents:
        store_parent_block(parent["parent_id"], parent)

    total_indexed = 0
    
    for i in range(0, len(children), BATCH_SIZE):
        batch = children[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed(texts)
        
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": vectors[j]},
                payload={
                    "text": batch[j]["text"],
                    "parent_id": batch[j]["parent_id"],
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


def retrieve_parent_blocks(parent_ids: list[str], parents_store: dict) -> list[dict]:

    seen = set()
    result = []
    
    for parent_id in parent_ids:
        if parent_id not in seen:
            parent = parents_store.get(parent_id)
            if parent:
                result.append(parent)
                seen.add(parent_id)
    
    return result


if __name__ == "__main__":
    import sys
    from src.pipeline.extraction import extract_pdf
    from src.chunking.chunking import chunk_pages
    from src.db.qdrant_setup import create_collection

    create_collection()

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"

    pages = extract_pdf(test_path)
    chunks = chunk_pages(pages)
    count = index_chunks(chunks)
    print(f"Baseline: Indexed {count} chunks")

    total_in_collection = client.count(COLLECTION_NAME).count
    print(f"Total points in collection '{COLLECTION_NAME}': {total_in_collection}")
