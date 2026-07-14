import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.embedding import embed, EMBEDDING_SIZE


class TestEmbed:

    def test_string_returns_vector_of_correct_size(self):
        vector = embed("test")
        assert isinstance(vector, list)
        assert len(vector) == EMBEDDING_SIZE
        assert all(isinstance(x, float) for x in vector)

    def test_list_of_strings_returns_list_of_vectors(self):
        vectors = embed(["a", "b"])
        assert isinstance(vectors, list)
        assert len(vectors) == 2
        for v in vectors:
            assert len(v) == EMBEDDING_SIZE

    def test_single_item_list_stays_a_list_of_vectors(self):
        vectors = embed(["text"])
        assert isinstance(vectors, list)
        assert len(vectors) == 1
        assert len(vectors[0]) == EMBEDDING_SIZE

    def test_same_text_gives_same_vector(self):
        v1 = embed("repeated text")
        v2 = embed("repeated text")
        assert v1 == v2

    def test_different_text_gives_different_vectors(self):
        v1 = embed("text about apples")
        v2 = embed("text about cars")
        assert v1 != v2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])