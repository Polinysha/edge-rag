from src.llm.llm_client import call_llm

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using ONLY "
    "the context provided below. Do not use any outside knowledge. "
    "If the context does not contain enough information to answer the question, "
    "say clearly that the answer is not in the provided context — do not guess "
    "or make anything up."
)


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}, page {c['page_num']}]\n{c['text']}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[dict]) -> str:

    context = build_context(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]

    return call_llm(messages, temperature=0.0)


if __name__ == "__main__":
    from src.pipeline.retrieval import search

    question = "what is this document about"
    chunks = search(question, top_k=5)
    answer = generate(question, chunks)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
