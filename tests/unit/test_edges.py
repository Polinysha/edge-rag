import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from langgraph.graph import END
from src.pipeline.edges import should_retrieve_again, should_retry


class TestShouldRetrieveAgain:

    def test_low_score_and_low_retry_returns_rewrite(self):
        state = {"context_score": 0.3, "retry_count": 0}
        assert should_retrieve_again(state) == "rewrite"

    def test_low_score_but_retry_exhausted_returns_generate(self):
        state = {"context_score": 0.3, "retry_count": 2}
        assert should_retrieve_again(state) == "generate"

    def test_high_score_returns_generate_regardless_of_retry(self):
        state = {"context_score": 0.8, "retry_count": 0}
        assert should_retrieve_again(state) == "generate"

    def test_exact_threshold_score_returns_generate(self):
        state = {"context_score": 0.5, "retry_count": 0}
        assert should_retrieve_again(state) == "generate"

    def test_retry_count_one_still_allows_rewrite(self):
        state = {"context_score": 0.3, "retry_count": 1}
        assert should_retrieve_again(state) == "rewrite"


class TestShouldRetry:

    def test_low_relevance_first_retry_returns_rewrite(self):
        state = {"context_relevance": 0.2, "retry_count": 0}
        assert should_retry(state) == "rewrite"

    def test_low_relevance_retry_count_one_returns_rewrite(self):
        state = {"context_relevance": 0.2, "retry_count": 1}
        assert should_retry(state) == "rewrite"

    def test_low_relevance_retry_exhausted_returns_end(self):
        state = {"context_relevance": 0.2, "retry_count": 2}
        assert should_retry(state) == END

    def test_high_relevance_returns_end(self):
        state = {"context_relevance": 0.8, "retry_count": 0}
        assert should_retry(state) == END

    def test_exact_threshold_relevance_returns_end(self):
        state = {"context_relevance": 0.4, "retry_count": 0}
        assert should_retry(state) == END

    def test_missing_context_relevance_defaults_to_end(self):
        state = {"retry_count": 0}
        assert should_retry(state) == END


if __name__ == "__main__":
    pytest.main([__file__, "-v"])