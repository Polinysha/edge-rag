import asyncio

from langsmith import Client as LangSmithClient
from langsmith import get_current_run_tree

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import Faithfulness, ContextRelevance, AnswerRelevancy

from src.llm.llm_client import async_client as openrouter_async_client, MODEL_NAME
from src.pipeline.state import State

MAX_RETRIES_PER_METRIC = 3

_ragas_llm = llm_factory(
    model=MODEL_NAME,
    provider="openai",
    client=openrouter_async_client,
    max_tokens=2000,
)
_ragas_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

_faithfulness_metric = Faithfulness(llm=_ragas_llm)
_context_relevance_metric = ContextRelevance(llm=_ragas_llm)
_answer_relevancy_metric = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings)


async def _score_with_retries(coro_factory, metric_name: str) -> float:

    last_error = None
    for attempt in range(MAX_RETRIES_PER_METRIC):
        try:
            result = await coro_factory()
            return result.value
        except Exception as e:
            last_error = e

    print(f"Warning: {metric_name} failed after {MAX_RETRIES_PER_METRIC} attempts "
          f"({last_error}). Falling back to 0.0.")
    return 0.0


async def _score_all(question: str, answer: str, contexts: list[str]) -> dict:
    faithfulness_value, context_relevance_value, answer_relevance_value = await asyncio.gather(
        _score_with_retries(
            lambda: _faithfulness_metric.ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            ),
            "faithfulness",
        ),
        _score_with_retries(
            lambda: _context_relevance_metric.ascore(
                user_input=question, retrieved_contexts=contexts
            ),
            "context_relevance",
        ),
        _score_with_retries(
            lambda: _answer_relevancy_metric.ascore(user_input=question, response=answer),
            "answer_relevance",
        ),
    )
    return {
        "faithfulness": faithfulness_value,
        "context_relevance": context_relevance_value,
        "answer_relevance": answer_relevance_value,
    }


def evaluate(state: State) -> dict:

    contexts = [doc["text"] for doc in state["documents"]]
    scores = asyncio.run(_score_all(state["question"], state["generation"], contexts))

    run_tree = get_current_run_tree()
    if run_tree is not None:
        ls_client = LangSmithClient()
        for metric_name, value in scores.items():
            ls_client.create_feedback(run_tree.id, key=metric_name, score=value)

    return scores

if __name__ == "__main__":
    from src.pipeline.embedding import embed
    from src.pipeline.retrieve_node import retrieve
    from src.pipeline.generate_node import generate

    question = "what is this document about"
    state = {"question": question, "query_vector": embed(question)}
    state.update(retrieve(state))
    state.update(generate(state))

    result = evaluate(state)
    print(f"Question: {question}")
    print(f"Scores: {result}")