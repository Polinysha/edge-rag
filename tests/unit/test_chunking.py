import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.chunking.chunking import chunk_pages, CHUNK_SIZE


def make_page(text: str, page_num: int = 1, source: str = "test.pdf") -> dict:
    return {"text": text, "page_num": page_num, "source": source}


class TestChunkPages:

    def test_does_not_exceed_chunk_size_with_large_margin(self):
        long_text = "word " * 500
        pages = [make_page(long_text)]
        chunks = chunk_pages(pages)

        for c in chunks:
            assert len(c["text"]) <= CHUNK_SIZE * 1.5

    def test_preserves_all_metadata(self):
        pages = [make_page("some page text", page_num=3, source="doc.pdf")]
        chunks = chunk_pages(pages)

        assert len(chunks) > 0
        for c in chunks:
            assert c["source"] == "doc.pdf"
            assert c["page_num"] == 3
            assert "chunk_idx" in c
            assert "text" in c

    def test_chunk_idx_increases_in_order(self):
        pages = [make_page("text " * 200, page_num=1)]
        chunks = chunk_pages(pages)

        indices = [c["chunk_idx"] for c in chunks]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices)))

    def test_chunk_count_grows_with_text_volume(self):
        short_pages = [make_page("short text")]
        long_pages = [make_page("long text " * 300)]

        short_chunks = chunk_pages(short_pages)
        long_chunks = chunk_pages(long_pages)

        assert len(long_chunks) > len(short_chunks)

    def test_empty_page_produces_no_chunks(self):
        pages = [make_page("")]
        chunks = chunk_pages(pages)
        assert len(chunks) == 0

    def test_multiple_pages_get_continuous_chunk_idx(self):
        pages = [
            make_page("first page text " * 50, page_num=1),
            make_page("second page text " * 50, page_num=2),
        ]
        chunks = chunk_pages(pages)

        page1_chunks = [c for c in chunks if c["page_num"] == 1]
        page2_chunks = [c for c in chunks if c["page_num"] == 2]

        assert len(page1_chunks) > 0
        assert len(page2_chunks) > 0
        assert page2_chunks[0]["chunk_idx"] > page1_chunks[-1]["chunk_idx"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])