from src.pipeline.generation import generate as generate_from_chunks
from src.pipeline.state import State


def generate(state: State) -> dict:

    answer = generate_from_chunks(state["question"], state["documents"])
    return {"generation": answer}


if __name__ == "__main__":
    from src.pipeline.embedding import embed
    from src.pipeline.retrieve_node import retrieve

    question = "what is this document about"
    state = {"question": question, "query_vector": embed(question)}
    state.update(retrieve(state))

    result = generate(state)
    print(f"Question: {question}")
    print(f"Generation: {result['generation']}")
