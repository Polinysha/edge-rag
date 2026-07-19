import sys
import os
import json
from unittest.mock import patch, AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.pipeline.expand_query import expand_query
from src.pipeline.retrieve_node import rrf_merge, retrieve
from src.pipeline.embedding import embed
from src.pipeline.grade_documents import grade_documents
from src.pipeline.rewrite_query import rewrite_query
from src.pipeline.generate_node import generate


class TestExpandQuery:

    def test_returns_three_rewrites_and_correct_vector_size(self):
        fake_rewrites = [
            "What duties do the contract parties have?",
            "What are the responsibilities of each party?",
            "What is required of the signatories?",
        ]
        fake_llm_response = json.dumps(fake_rewrites)

        with patch("src.pipeline.expand_query.call_llm", return_value=fake_llm_response):
            result = expand_query({"question": "What are the parties' obligations?"})

        assert len(result["rewrites"]) == 3
        assert result["rewrites"] == fake_rewrites
        assert len(result["query_vector"]) == 384

    def test_query_vector_is_a_list_of_floats(self):
        fake_rewrites = ["a", "b", "c"]
        fake_llm_response = json.dumps(fake_rewrites)

        with patch("src.pipeline.expand_query.call_llm", return_value=fake_llm_response):
            result = expand_query({"question": "test question"})

        assert isinstance(result["query_vector"], list)
        assert all(isinstance(x, float) for x in result["query_vector"])

    def test_does_not_call_real_llm(self):
        fake_rewrites = ["x", "y", "z"]
        with patch("src.pipeline.expand_query.call_llm", return_value=json.dumps(fake_rewrites)) as mock_llm:
            expand_query({"question": "anything"})
            mock_llm.assert_called_once()

class TestRrfMerge:

    def test_matches_hand_computed_ranking(self):
        chunk_a = {"source": "doc", "page_num": 1, "chunk_idx": 0, "label": "A"}
        chunk_b = {"source": "doc", "page_num": 1, "chunk_idx": 1, "label": "B"}
        chunk_c = {"source": "doc", "page_num": 1, "chunk_idx": 2, "label": "C"}
        chunk_d = {"source": "doc", "page_num": 1, "chunk_idx": 3, "label": "D"}

        dense_results = [chunk_a, chunk_b, chunk_c]
        sparse_results = [chunk_a, chunk_d, chunk_b]

        merged = rrf_merge(dense_results, sparse_results, k=60)

        labels_in_order = [c["label"] for c in merged]
        assert labels_in_order == ["A", "B", "D", "C"]

    def test_chunk_present_in_both_lists_ranks_above_single_list_chunk(self):
        chunk_in_both = {"source": "doc", "page_num": 1, "chunk_idx": 0, "label": "both"}
        chunk_dense_only = {"source": "doc", "page_num": 1, "chunk_idx": 1, "label": "dense_only"}

        dense_results = [chunk_in_both, chunk_dense_only]
        sparse_results = [chunk_in_both]

        merged = rrf_merge(dense_results, sparse_results)
        assert merged[0]["label"] == "both"

    def test_empty_inputs_return_empty_list(self):
        assert rrf_merge([], []) == []


class TestRetrieve:

    def test_finds_known_relevant_chunk(self):
        question = "internship agreement organization"
        state = {"question": question, "query_vector": embed(question)}

        result = retrieve(state)

        assert len(result["documents"]) > 0
        assert len(result["documents"]) <= 5
        texts = " ".join(doc["text"] for doc in result["documents"]).lower()
        assert "практик" in texts or "договор" in texts


class TestGradeDocuments:

    def test_context_score_matches_known_yes_no_proportion(self):
        # 3 chunks: 2 "yes", 1 "no" -> expected context_score = 2/3
        documents = [
            {"text": "relevant chunk 1"},
            {"text": "relevant chunk 2"},
            {"text": "irrelevant chunk"},
        ]
        state = {"question": "test question", "documents": documents, "retry_count": 0}

        responses = iter(["yes", "yes", "no"])
        with patch("src.pipeline.grade_documents.call_llm", side_effect=lambda *a, **k: next(responses)):
            result = grade_documents(state)

        assert result["context_score"] == pytest.approx(2 / 3)

    def test_all_yes_gives_score_of_one(self):
        documents = [{"text": "a"}, {"text": "b"}]
        state = {"question": "q", "documents": documents, "retry_count": 0}

        with patch("src.pipeline.grade_documents.call_llm", return_value="yes"):
            result = grade_documents(state)

        assert result["context_score"] == 1.0

    def test_all_no_gives_score_of_zero(self):
        documents = [{"text": "a"}, {"text": "b"}]
        state = {"question": "q", "documents": documents, "retry_count": 0}

        with patch("src.pipeline.grade_documents.call_llm", return_value="no"):
            result = grade_documents(state)

        assert result["context_score"] == 0.0

    def test_empty_documents_gives_score_of_zero_without_calling_llm(self):
        state = {"question": "q", "documents": [], "retry_count": 0}

        with patch("src.pipeline.grade_documents.call_llm") as mock_llm:
            result = grade_documents(state)
            mock_llm.assert_not_called()

        assert result["context_score"] == 0.0


class TestRewriteQuery:

    def test_retry_count_increments_by_exactly_one(self):
        state = {"question": "original question", "retry_count": 0}

        with patch("src.pipeline.rewrite_query.call_llm", return_value="rewritten question"):
            result = rewrite_query(state)

        assert result["retry_count"] == 1

    def test_question_changes_from_original(self):
        state = {"question": "original question", "retry_count": 0}

        with patch("src.pipeline.rewrite_query.call_llm", return_value="a more specific question"):
            result = rewrite_query(state)

        assert result["question"] != "original question"
        assert result["question"] == "a more specific question"

    def test_retry_count_increments_from_nonzero_starting_value(self):
        state = {"question": "q", "retry_count": 1}

        with patch("src.pipeline.rewrite_query.call_llm", return_value="q2"):
            result = rewrite_query(state)

        assert result["retry_count"] == 2


class TestGenerateNode:

    def test_says_no_answer_when_context_is_unrelated(self):
        unrelated_chunks = [
            {"text": "The recipe calls for flour, eggs, and salt.", "source": "r.pdf", "page_num": 1, "chunk_idx": 0},
        ]
        state = {"question": "What is the capital of France?", "documents": unrelated_chunks}

        fake_no_answer = "The provided context does not contain information about this."
        with patch("src.pipeline.generation.call_llm", return_value=fake_no_answer):
            result = generate(state)

        assert "generation" in result
        assert "does not contain" in result["generation"].lower()

    def test_returns_nonempty_answer_with_relevant_context(self):
        relevant_chunks = [
            {"text": "This is an internship agreement contract.", "source": "doc.pdf", "page_num": 1, "chunk_idx": 0},
        ]
        state = {"question": "What is this document about?", "documents": relevant_chunks}

        fake_answer = "This document is an internship agreement contract."
        with patch("src.pipeline.generation.call_llm", return_value=fake_answer):
            result = generate(state)

        assert len(result["generation"]) > 0
        assert result["generation"] == fake_answer


class TestEvaluateNode:

    def _make_metric_result(self, value):
        fake_result = MagicMock()
        fake_result.value = value
        return fake_result

    def test_all_three_metrics_present_with_mocked_ragas(self):
        from src.pipeline.evaluate_node import evaluate

        state = {
            "question": "what is this document about",
            "documents": [{"text": "some chunk text"}],
            "generation": "some answer",
        }

        with patch(
            "src.pipeline.evaluate_node._faithfulness_metric.ascore",
            new=AsyncMock(return_value=self._make_metric_result(0.9)),
        ), patch(
            "src.pipeline.evaluate_node._context_relevance_metric.ascore",
            new=AsyncMock(return_value=self._make_metric_result(0.8)),
        ), patch(
            "src.pipeline.evaluate_node._answer_relevancy_metric.ascore",
            new=AsyncMock(return_value=self._make_metric_result(0.7)),
        ), patch(
            "src.pipeline.evaluate_node.get_current_run_tree", return_value=None,
        ):
            result = evaluate(state)

        assert result["faithfulness"] == 0.9
        assert result["context_relevance"] == 0.8
        assert result["answer_relevance"] == 0.7

    def test_failing_metric_falls_back_to_zero_without_crashing(self):
        from src.pipeline.evaluate_node import evaluate

        state = {
            "question": "q",
            "documents": [{"text": "chunk"}],
            "generation": "answer",
        }

        always_fails = AsyncMock(side_effect=ValueError("simulated malformed JSON"))

        with patch(
            "src.pipeline.evaluate_node._faithfulness_metric.ascore", new=always_fails,
        ), patch(
            "src.pipeline.evaluate_node._context_relevance_metric.ascore",
            new=AsyncMock(return_value=self._make_metric_result(0.5)),
        ), patch(
            "src.pipeline.evaluate_node._answer_relevancy_metric.ascore",
            new=AsyncMock(return_value=self._make_metric_result(0.5)),
        ), patch(
            "src.pipeline.evaluate_node.get_current_run_tree", return_value=None,
        ):
            result = evaluate(state)

        assert result["faithfulness"] == 0.0
        assert result["context_relevance"] == 0.5
        assert result["answer_relevance"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])