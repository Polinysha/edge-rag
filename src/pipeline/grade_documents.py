from src.llm.llm_client import call_llm
from src.pipeline.state import State

GRADE_PROMPT = (
    "Does the following text help answer the question? Answer with exactly one "
    "word: yes or no.\n\n"
    "Question: {question}\n\n"
    "Text: {text}"
)


def grade_one_chunk(question: str, chunk_text: str) -> bool:
    prompt = GRADE_PROMPT.format(question=question, text=chunk_text)
    response = call_llm([{"role": "user", "content": prompt}])
    return response.strip().lower().startswith("yes")


def grade_documents(state: State) -> dict:

    documents = state["documents"]

    if not documents:
        return {"context_score": 0.0, "retry_count": state.get("retry_count", 0)}

    grades = [grade_one_chunk(state["question"], doc["text"]) for doc in documents]
    context_score = sum(grades) / len(grades)

    return {"context_score": context_score, "retry_count": state.get("retry_count", 0)}


if __name__ == "__main__":
    from src.pipeline.embedding import embed
    from src.pipeline.retrieve_node import retrieve

    question = "what is this document about"
    state = {"question": question, "query_vector": embed(question), "retry_count": 0}
    state.update(retrieve(state))

    result = grade_documents(state)
    print(f"Question: {question}")
    print(f"context_score: {result['context_score']}")
