from functools import lru_cache

from langgraph.graph import END, StateGraph

from src.pipeline.state import State
from src.pipeline.expand_query import expand_query
from src.pipeline.retrieve_node import retrieve
from src.pipeline.grade_documents import grade_documents
from src.pipeline.rewrite_query import rewrite_query
from src.pipeline.generate_node import generate
from src.pipeline.evaluate_node import evaluate
from src.pipeline.edges import should_retrieve_again, should_retry


def build_graph():
    builder = StateGraph(State)

    builder.add_node("expand_query", expand_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)

    builder.set_entry_point("expand_query")
    builder.add_edge("expand_query", "retrieve")
    builder.add_edge("retrieve", "grade_documents")

    builder.add_conditional_edges(
        "grade_documents",
        should_retrieve_again,
        {"rewrite": "rewrite_query", "generate": "generate"},
    )

    builder.add_edge("rewrite_query", "expand_query")

    builder.add_edge("generate", "evaluate")

    builder.add_conditional_edges(
        "evaluate",
        should_retry,
        {"rewrite": "rewrite_query", END: END},
    )

    return builder.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "what is this document about"
    graph = build_graph()
    final_state = graph.invoke({"question": question, "retry_count": 0})

    print(f"Question:  {question}")
    print(f"Answer:    {final_state.get('generation')}")
    print(f"Retries:   {final_state.get('retry_count')}")
    print(f"Metrics:   context_relevance={final_state.get('context_relevance')} "
          f"faithfulness={final_state.get('faithfulness')} "
          f"answer_relevance={final_state.get('answer_relevance')}")
