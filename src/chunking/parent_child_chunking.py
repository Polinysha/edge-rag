from langchain_text_splitters import RecursiveCharacterTextSplitter

PARENT_CHUNK_SIZE = 500
PARENT_CHUNK_OVERLAP = 50

CHILD_CHUNK_SIZE = 150
CHILD_CHUNK_OVERLAP = 30

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE,
    chunk_overlap=PARENT_CHUNK_OVERLAP,
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE,
    chunk_overlap=CHILD_CHUNK_OVERLAP,
)


def create_parent_child_chunks(pages: list[dict]) -> tuple[list[dict], list[dict]]:

    parents = []
    children = []
    parent_id_counter = 0
    child_idx = 0
    
    for page in pages:
        page_parent_texts = parent_splitter.split_text(page["text"])
        
        for parent_text in page_parent_texts:
            if not parent_text.strip():
                continue
            
            parent_id = f"parent_{parent_id_counter}"
            parent_id_counter += 1

            parents.append({
                "parent_id": parent_id,
                "text": parent_text,
                "source": page["source"],
                "page_num": page["page_num"],
            })

            child_texts = child_splitter.split_text(parent_text)
            
            for child_text in child_texts:
                if not child_text.strip():
                    continue
                
                children.append({
                    "text": child_text,
                    "parent_id": parent_id,
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_idx": child_idx,
                })
                child_idx += 1
    
    return parents, children


def get_parent_blocks(children: list[dict], parents: list[dict]) -> list[dict]:

    if not children or not parents:
        return []

    parent_lookup = {}
    for parent in parents:
        parent_lookup[parent["parent_id"]] = parent

    seen_parent_ids = set()
    result = []
    
    for child in children:
        parent_id = child.get("parent_id")
        if parent_id and parent_id not in seen_parent_ids:
            parent = parent_lookup.get(parent_id)
            if parent:
                result.append(parent)
                seen_parent_ids.add(parent_id)
    
    return result

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../pipeline")
    from src.pipeline.extraction import extract_pdf
    
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"
    pages = extract_pdf(test_path)
    
    parents, children = create_parent_child_chunks(pages)
    
    print(f"Pages: {len(pages)}")
    print(f"Parent blocks: {len(parents)}")
    print(f"Child chunks: {len(children)}")
    print()
    
    if parents:
        print("=== First parent block ===")
        print(f"parent_id: {parents[0]['parent_id']}")
        print(f"text length: {len(parents[0]['text'])}")
        print(parents[0]['text'][:200])
        print()
    
    if children:
        print("=== First 3 child chunks ===")
        for child in children[:3]:
            print(f"chunk_idx: {child['chunk_idx']}, parent_id: {child['parent_id']}")
            print(f"text length: {len(child['text'])}")
            print(child['text'][:100])
            print()
