"""
Юнит-тесты на embedding.py.

Запуск:
    uv run pytest tests/unit/test_embedding.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipeline.embedding import embed, EMBEDDING_SIZE


class TestEmbed:

    def test_строка_возвращает_вектор_правильной_размерности(self):
        vector = embed("тест")
        assert isinstance(vector, list)
        assert len(vector) == EMBEDDING_SIZE
        assert all(isinstance(x, float) for x in vector)

    def test_список_строк_возвращает_список_векторов(self):
        vectors = embed(["a", "b"])
        assert isinstance(vectors, list)
        assert len(vectors) == 2
        for v in vectors:
            assert len(v) == EMBEDDING_SIZE

    def test_список_из_одной_строки_тоже_список_векторов(self):
        # граничный случай: ['текст'] должно остаться списком векторов,
        # а не схлопнуться в одиночный вектор, как при передаче просто строки
        vectors = embed(["текст"])
        assert isinstance(vectors, list)
        assert len(vectors) == 1
        assert len(vectors[0]) == EMBEDDING_SIZE

    def test_одинаковый_текст_даёт_одинаковый_вектор(self):
        v1 = embed("повторяющийся текст")
        v2 = embed("повторяющийся текст")
        assert v1 == v2

    def test_разный_текст_даёт_разные_векторы(self):
        v1 = embed("текст про яблоки")
        v2 = embed("текст про автомобили")
        assert v1 != v2


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
