from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_SIZE = 384

_model = SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: str | list[str]) -> list[float] | list[list[float]]:

    is_single = isinstance(texts, str)
    input_texts = [texts] if is_single else texts

    vectors = _model.encode(input_texts, convert_to_numpy=True).tolist()

    return vectors[0] if is_single else vectors


if __name__ == "__main__":
    v1 = embed("test")
    print(f"embed('test') -> vector of length {len(v1)}")

    v2 = embed(["a", "b"])
    print(f"embed(['a', 'b']) -> {len(v2)} vectors, each of length {len(v2[0])}")