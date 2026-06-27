"""
Разбиение извлечённого текста на чанки фиксированного размера (baseline)
с сохранением метаданных: source, page_num, chunk_idx.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Принимает список страниц от extract_pdf: [{page_num, text, source}, ...]
    Возвращает список чанков: [{text, source, page_num, chunk_idx}, ...]
    """
    chunks = []
    chunk_idx = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            if not chunk_text.strip():
                continue
            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page_num": page["page_num"],
                "chunk_idx": chunk_idx,
            })
            chunk_idx += 1

    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.pipeline.extraction import extract_pdf

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"
    pages = extract_pdf(test_path)
    chunks = chunk_pages(pages)

    print(f"Страниц: {len(pages)}, чанков: {len(chunks)}")
    print()
    for c in chunks[:3]:
        print(f"--- chunk_idx={c['chunk_idx']}, page_num={c['page_num']}, source={c['source']} ---")
        print(f"длина текста: {len(c['text'])}")
        print(c["text"][:150])
        print()