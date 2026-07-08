import json
import numpy as np

from src.llm.llm_client import call_llm
from src.pipeline.embedding import embed
from src.pipeline.state import State

EXPAND_PROMPT = (
    "Generate 3 alternative phrasings of the following question. "
    "They should ask for the same information using different words. "
    "Respond ONLY with a JSON array of 3 strings, nothing else.\n\n"
    "Question: {question}"
)


def expand_query(state: State) -> dict:

    prompt = EXPAND_PROMPT.format(question=state["question"])
    raw_response = call_llm([{"role": "user", "content": prompt}])

    rewrites = json.loads(raw_response)

    all_texts = [state["question"]] + rewrites
    vectors = embed(all_texts)
    query_vector = np.mean(vectors, axis=0).tolist()

    return {"rewrites": rewrites, "query_vector": query_vector}


if __name__ == "__main__":
    result = expand_query({"question": "What are the parties' obligations?"})
    print("Rewrites:", result["rewrites"])
    print("Query vector length:", len(result["query_vector"]))
