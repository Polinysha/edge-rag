from typing import TypedDict


class State(TypedDict):
    question: str
    rewrites: list[str]
    query_vector: list[float]
    documents: list[dict]
    generation: str
    context_score: float
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    retry_count: int
