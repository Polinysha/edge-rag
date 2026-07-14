import sys
import os
from io import BytesIO
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealth:

    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_shape(self):
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert "version" in body
        assert body["status"] == "ok"


class TestUpload:

    def test_returns_200_with_mocked_pipeline(self):
        fake_pages = [{"page_num": 1, "text": "some text", "source": "fake.pdf"}]
        fake_chunks = [{"text": "some text", "source": "fake.pdf", "page_num": 1, "chunk_idx": 0}]

        with patch("src.api.main.extract_pdf", return_value=fake_pages), \
             patch("src.api.main.chunk_pages", return_value=fake_chunks), \
             patch("src.api.main.index_chunks", return_value=1):

            fake_file = BytesIO(b"%PDF-1.4 fake content")
            response = client.post(
                "/upload",
                files={"file": ("fake.pdf", fake_file, "application/pdf")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["filename"] == "fake.pdf"
        assert body["pages"] == 1
        assert body["chunks_indexed"] == 1

    def test_missing_file_returns_422(self):
        response = client.post("/upload")
        assert response.status_code == 422


class TestAsk:

    def test_returns_200_with_mocked_graph(self):
        fake_state = {
            "generation": "This is a mocked answer.",
            "documents": [
                {"source": "fake.pdf", "page_num": 1, "text": "...", "chunk_idx": 0}
            ],
            "context_relevance": 0.9,
            "faithfulness": 0.8,
            "answer_relevance": 0.7,
        }

        class FakeGraph:
            def invoke(self, _inputs):
                return fake_state

        with patch("src.pipeline.graph.get_graph", return_value=FakeGraph()):
            response = client.post("/ask", json={"question": "any question"})

        assert response.status_code == 200
        body = response.json()
        assert "answer" in body
        assert "sources" in body
        assert "metrics" in body
        assert body["answer"] == "This is a mocked answer."
        assert body["sources"] == [{"source": "fake.pdf", "page_num": 1}]
        assert body["metrics"]["faithfulness"] == 0.8

    def test_missing_question_field_returns_422(self):
        response = client.post("/ask", json={})
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
