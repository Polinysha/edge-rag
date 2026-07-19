import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.pipeline.graph import build_graph

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "thresholds.json")
if os.path.exists(THRESHOLDS_PATH):
    with open(THRESHOLDS_PATH) as f:
        THRESHOLDS = json.load(f)
else:
    THRESHOLDS = {
        "context_relevance": 0.5,
        "faithfulness": 0.25,
        "answer_relevance": 0.1,
    }

GOLDEN = [
    {"question": "Who are the parties to the agreement?", "expect": ["part"]},
    {"question": "What must be prepared at the end of the internship?", "expect": ["report"]},
    {"question": "What is the duration of the contract?", "expect": ["month", "year", "day"]},
    {"question": "What is the penalty for late delivery?", "expect": ["%", "penalt", "fee"]},
    # intentionally vague phrasing -> should trigger a rewrite before answering
    {"question": "how long does it all last", "expect": ["month", "year", "day"]},
]

PASS_THRESHOLD = 0.8

@pytest.mark.slow
def test_graph_answers_golden_dataset():
    graph = build_graph()

    correct = 0
    misses = []
    for item in GOLDEN:
        state = graph.invoke({"question": item["question"], "retry_count": 0})
        answer = (state.get("generation") or "").lower()
        if any(kw.lower() in answer for kw in item["expect"]):
            correct += 1
        else:
            misses.append((item["question"], answer[:120]))

    ratio = correct / len(GOLDEN)
    assert ratio >= PASS_THRESHOLD, (
        f"only {correct}/{len(GOLDEN)} golden questions answered correctly "
        f"(threshold {PASS_THRESHOLD:.0%}). Misses: {misses}"
    )


@pytest.mark.slow
def test_graph_populates_full_state():
    graph = build_graph()
    state = graph.invoke(
        {"question": GOLDEN[0]["question"], "retry_count": 0}
    )

    assert state.get("generation")
    assert state.get("documents")
    for metric in ("context_relevance", "faithfulness", "answer_relevance"):
        assert metric in state
        assert isinstance(state[metric], (int, float))


@pytest.mark.slow
def test_graph_metrics_above_thresholds():
    graph = build_graph()
    state = graph.invoke({"question": GOLDEN[0]["question"], "retry_count": 0})

    for metric, threshold in THRESHOLDS.items():
        actual = state.get(metric, 0.0)
        assert actual >= threshold - 0.05, (
            f"{metric} = {actual:.3f} below threshold {threshold:.3f} "
            f"(tolerance 0.05). Possible regression."
        )

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
