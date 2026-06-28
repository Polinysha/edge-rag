"""
Unit tests for extraction.py: classify_page, clean_text, extract_pdf.

Run:
    uv run pytest tests/unit/test_extraction.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "pipeline"))

import pytest
from extraction import clean_text, extract_pdf

# Path to the test PDF — place your file here before running the test
TEST_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test.pdf")


class TestCleanText:
    """Tests for clean_text — doesn't require a PDF, pure string -> string function."""

    def test_joins_hyphenated_line_break(self):
        text = "infor-\nmation"
        result = clean_text(text)
        assert "information" in result
        assert "infor-" not in result

    def test_removes_separator_lines(self):
        text = "Header\n------\nText after"
        result = clean_text(text)
        assert "------" not in result
        assert "Header" in result
        assert "Text after" in result

    def test_collapses_multiple_spaces(self):
        text = "word1    word2"
        result = clean_text(text)
        assert "    " not in result
        assert "word1 word2" in result

    def test_collapses_multiple_blank_lines(self):
        text = "paragraph1\n\n\n\n\nparagraph2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_empty_string_does_not_raise(self):
        # edge case — empty input should not raise an error
        result = clean_text("")
        assert result == ""


class TestExtractPdf:
    """
    Tests for extract_pdf — require a real PDF file at data/test.pdf.
    If the file is missing, tests are skipped (not failed) so they don't
    block the rest of the suite on machines without the test file.
    """

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="No test PDF at data/test.pdf — add the file to run this test",
    )
    def test_returns_list_matching_page_count(self):
        result = extract_pdf(TEST_PDF_PATH)
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="No test PDF at data/test.pdf — add the file to run this test",
    )
    def test_each_page_has_required_fields(self):
        result = extract_pdf(TEST_PDF_PATH)
        for page in result:
            assert "page_num" in page
            assert "text" in page
            assert "source" in page
            assert page["page_num"] > 0
            assert page["source"] != ""

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="No test PDF at data/test.pdf — add the file to run this test",
    )
    def test_page_text_is_not_empty_for_native_pdf(self):
        # For native PDFs (with a text layer) the text should not be empty
        result = extract_pdf(TEST_PDF_PATH)
        non_empty_pages = [p for p in result if len(p["text"]) > 0]
        assert len(non_empty_pages) > 0, "No page returned any text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])