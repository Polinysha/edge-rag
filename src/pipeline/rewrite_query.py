from src.llm.llm_client import call_llm
from src.pipeline.state import State

REWRITE_PROMPT = (
    "Rephrase the following question to be more specific and use more precise "
    "terminology, so it is more likely to match the wording used in a document. "
    "Respond with ONLY the rephrased question, nothing else.\n\n"
    "Question: {question}"
)


def rewrite_query(state: State) -> dict:

    prompt = REWRITE_PROMPT.format(question=state["question"])
    new_question = call_llm([{"role": "user", "content": prompt}]).strip()

    return {
        "question": new_question,
        "retry_count": state.get("retry_count", 0) + 1,
    }


if __name__ == "__main__":
    original = {"question": "What is this document about?", "retry_count": 0}
    result = rewrite_query(original)
    print(f"Original question: {original['question']}")
    print(f"Rewritten question: {result['question']}")
    print(f"retry_count: {result['retry_count']}")
