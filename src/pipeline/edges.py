from langgraph.graph import END

from src.pipeline.state import State


def should_retrieve_again(state: State) -> str:

    if state["context_score"] < 0.5 and state["retry_count"] < 2:
        return "rewrite"
    return "generate"


def should_retry(state: State) -> str:

    if state.get("context_relevance", 1.0) < 0.4 and state.get("retry_count", 0) <= 1:
        return "rewrite"
    return END
