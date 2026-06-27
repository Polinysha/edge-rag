"""
Юнит-тесты на chunking.py.

Запуск:
    uv run pytest tests/unit/test_chunking.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.chunking import chunk_pages, CHUNK_SIZE


def make_page(text: str, page_num: int = 1, source: str = "test.pdf") -> dict:
    return {"text": text, "page_num": page_num, "source": source}


class TestChunkPages:

    def test_не_превышает_chunk_size_с_большим_запасом(self):
        long_text = "слово " * 500  # заведомо длинный текст
        pages = [make_page(long_text)]
        chunks = chunk_pages(pages)

        for c in chunks:
            # допускаем небольшой запас сверху, но не больше чем в 1.5 раза
            assert len(c["text"]) <= CHUNK_SIZE * 1.5

    def test_сохраняет_все_метаданные(self):
        pages = [make_page("какой-то текст страницы", page_num=3, source="doc.pdf")]
        chunks = chunk_pages(pages)

        assert len(chunks) > 0
        for c in chunks:
            assert c["source"] == "doc.pdf"
            assert c["page_num"] == 3
            assert "chunk_idx" in c
            assert "text" in c

    def test_chunk_idx_растёт_по_порядку(self):
        pages = [make_page("текст " * 200, page_num=1)]
        chunks = chunk_pages(pages)

        indices = [c["chunk_idx"] for c in chunks]
        assert indices == sorted(indices)
        assert indices == list(range(len(indices)))

    def test_число_чанков_растёт_с_объёмом_текста(self):
        short_pages = [make_page("короткий текст")]
        long_pages = [make_page("длинный текст " * 300)]

        short_chunks = chunk_pages(short_pages)
        long_chunks = chunk_pages(long_pages)

        assert len(long_chunks) > len(short_chunks)

    def test_пустая_страница_не_создаёт_чанков(self):
        pages = [make_page("")]
        chunks = chunk_pages(pages)
        assert len(chunks) == 0

    def test_несколько_страниц_дают_сквозную_нумерацию_chunk_idx(self):
        pages = [
            make_page("текст первой страницы " * 50, page_num=1),
            make_page("текст второй страницы " * 50, page_num=2),
        ]
        chunks = chunk_pages(pages)

        page1_chunks = [c for c in chunks if c["page_num"] == 1]
        page2_chunks = [c for c in chunks if c["page_num"] == 2]

        assert len(page1_chunks) > 0
        assert len(page2_chunks) > 0
        # chunk_idx сквозной, не сбрасывается на новой странице
        assert page2_chunks[0]["chunk_idx"] > page1_chunks[-1]["chunk_idx"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
