import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from src.pipeline.generation import generate

UNRELATED_CHUNKS = [
    {
        "text": "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
        "source": "recipe.pdf",
        "page_num": 1,
        "chunk_idx": 0,
    },
    {
        "text": "Bake at 180 degrees Celsius for 25 minutes until golden brown.",
        "source": "recipe.pdf",
        "page_num": 1,
        "chunk_idx": 1,
    },
]

NO_ANSWER_MARKERS = [
    "not in the provided context",
    "does not contain",
    "doesn't contain",
    "no information",
    "not contain enough information",
    "cannot answer",
    "can't answer",
    "not mentioned",
    "not stated",
]

MAX_ATTEMPTS = 3


def contains_no_answer_marker(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in NO_ANSWER_MARKERS)


class TestGenerateContextDiscipline:

    def test_says_no_answer_when_context_is_unrelated(self):
        question = "What is the capital of France?"

        last_answer = None
        for attempt in range(MAX_ATTEMPTS):
            last_answer = generate(question, UNRELATED_CHUNKS)
            if contains_no_answer_marker(last_answer):
                return  # passed

        pytest.fail(
            f"generate() did not signal a missing answer after {MAX_ATTEMPTS} attempts. "
            f"Last answer: {last_answer!r}"
        )

    def test_does_not_invent_unrelated_facts(self):

        question = "What is the capital of France?"
        answer = generate(question, UNRELATED_CHUNKS)
        assert "paris" not in answer.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
