"""
Превращение текста в вектор (embedding) через sentence-transformers/all-MiniLM-L6-v2.
Модель загружается один раз при импорте модуля, не на каждый вызов.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

_model = SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: str | list[str]) -> list[float] | list[list[float]]:
    """
    Принимает строку или список строк, возвращает вектор(ы) размерности 384.
    Для одной строки — список из 384 float. Для списка строк — список таких списков.
    """
    is_single = isinstance(texts, str)
    input_texts = [texts] if is_single else texts

    vectors = _model.encode(input_texts, convert_to_numpy=True).tolist()

    return vectors[0] if is_single else vectors


if __name__ == "__main__":
    v1 = embed("тест")
    print(f"embed('тест') -> вектор длины {len(v1)}")

    v2 = embed(["a", "b"])
    print(f"embed(['a', 'b']) -> {len(v2)} векторов, каждый длины {len(v2[0])}")
