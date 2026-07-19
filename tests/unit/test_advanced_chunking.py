from src.chunking.parent_child_chunking import (
    create_parent_child_chunks,
    get_parent_blocks,
)
from src.pipeline.parent_store import (
    store_parent_block,
    get_parent_block,
    get_parent_blocks,
    clear_parent_store,
    get_store_size,
)


class TestParentChildChunking:
    def test_creates_parents_and_children(self):
        pages = [
            {
                "text": "This is a test document. " * 50,
                "source": "test.pdf",
                "page_num": 0,
            }
        ]

        parents, children = create_parent_child_chunks(pages)

        assert len(parents) > 0, "Should create at least one parent"
        assert len(children) > 0, "Should create at least one child"
        assert len(children) >= len(parents), "More children than parents"

    def test_each_child_has_parent_id(self):
        pages = [
            {
                "text": "Test content. " * 100,
                "source": "test.pdf",
                "page_num": 0,
            }
        ]

        parents, children = create_parent_child_chunks(pages)

        for child in children:
            assert "parent_id" in child, f"Child missing parent_id: {child}"
            assert child["parent_id"], "parent_id should not be empty"

    def test_child_text_is_substring_of_parent(self):
        pages = [
            {
                "text": "This is test content for chunking. " * 30,
                "source": "test.pdf",
                "page_num": 0,
            }
        ]

        parents, children = create_parent_child_chunks(pages)

        parent_lookup = {p["parent_id"]: p for p in parents}

        for child in children:
            parent_id = child["parent_id"]
            parent = parent_lookup.get(parent_id)
            assert parent, f"Parent {parent_id} not found"
            assert child["text"] in parent["text"], (
                f"Child text not in parent: '{child['text'][:50]}...' "
                f"not in '{parent['text'][:50]}...'"
            )

    def test_parent_blocks_preserve_metadata(self):
        pages = [
            {
                "text": "Content for page 0. " * 50,
                "source": "doc1.pdf",
                "page_num": 0,
            },
            {
                "text": "Content for page 1. " * 50,
                "source": "doc1.pdf",
                "page_num": 1,
            },
        ]

        parents, children = create_parent_child_chunks(pages)

        for parent in parents:
            assert "parent_id" in parent
            assert "text" in parent
            assert "source" in parent
            assert "page_num" in parent

    def test_get_parent_blocks_returns_unique_blocks(self):
        clear_parent_store()
        store_parent_block(
            "parent_0",
            {"parent_id": "parent_0", "text": "Parent 0 text", "page_num": 0},
        )
        store_parent_block(
            "parent_1",
            {"parent_id": "parent_1", "text": "Parent 1 text", "page_num": 1},
        )

        parent_ids = ["parent_0", "parent_1", "parent_0", "parent_1"]
        blocks = get_parent_blocks(parent_ids)

        assert len(blocks) == 2, "Should deduplicate"
        assert blocks[0]["parent_id"] == "parent_0"
        assert blocks[1]["parent_id"] == "parent_1"

    def test_empty_pages_produce_no_chunks(self):
        pages = [{"text": "", "source": "empty.pdf", "page_num": 0}]

        parents, children = create_parent_child_chunks(pages)

        assert len(parents) == 0
        assert len(children) == 0


class TestParentStore:

    def setup_method(self):
        clear_parent_store()

    def test_store_and_retrieve_parent(self):
        parent = {
            "parent_id": "test_parent",
            "text": "Test parent text",
            "page_num": 0,
        }

        store_parent_block("test_parent", parent)
        retrieved = get_parent_block("test_parent")

        assert retrieved == parent

    def test_get_nonexistent_parent_returns_none(self):
        result = get_parent_block("nonexistent")
        assert result is None

    def test_clear_parent_store(self):
        store_parent_block("p1", {"parent_id": "p1", "text": "Text 1"})
        store_parent_block("p2", {"parent_id": "p2", "text": "Text 2"})

        assert get_store_size() == 2

        clear_parent_store()

        assert get_store_size() == 0
        assert get_parent_block("p1") is None
        assert get_parent_block("p2") is None

    def test_get_parent_blocks_preserves_order(self):
        store_parent_block("p0", {"parent_id": "p0", "text": "Text 0"})
        store_parent_block("p1", {"parent_id": "p1", "text": "Text 1"})
        store_parent_block("p2", {"parent_id": "p2", "text": "Text 2"})

        parent_ids = ["p2", "p0", "p1", "p2"]
        blocks = get_parent_blocks(parent_ids)

        assert len(blocks) == 3
        assert blocks[0]["parent_id"] == "p2"
        assert blocks[1]["parent_id"] == "p0"
        assert blocks[2]["parent_id"] == "p1"


class TestStructuralChunking:
    def test_fallback_on_empty_pages(self):
        from src.chunking.structural_chunking import chunk_with_structure

        pages = []
        chunks = chunk_with_structure(pages, pdf_path=None)

        assert chunks == []

    def test_chunks_have_section_metadata(self):
        from src.chunking.structural_chunking import chunk_with_structure

        pages = [
            {
                "text": "This is test content. " * 100,
                "source": "test.pdf",
                "page_num": 0,
            }
        ]

        chunks = chunk_with_structure(pages, pdf_path=None)

        assert len(chunks) > 0
        for chunk in chunks:
            assert "section_title" in chunk

    def test_no_pdf_path_uses_single_section(self):
        from src.chunking.structural_chunking import chunk_with_structure

        pages = [
            {
                "text": "Content without structure. " * 50,
                "source": "test.pdf",
                "page_num": 0,
            }
        ]

        chunks = chunk_with_structure(pages, pdf_path=None)

        for chunk in chunks:
            assert chunk["section_title"] == "Document"
