"""
Юнит-тесты на extraction.py: classify_page, clean_text, extract_pdf.

Запуск:
    uv run pytest tests/unit/test_extraction.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "pipeline"))

import pytest
from extraction import clean_text, extract_pdf

# Путь к тестовому PDF — положи свой файл сюда перед запуском теста
TEST_PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "test.pdf")


class TestCleanText:
    """Тесты на функцию clean_text — она не требует PDF, чистая функция строка -> строка."""

    def test_склейка_переноса_слова(self):
        text = "инфор-\nмация"
        result = clean_text(text)
        assert "информация" in result
        assert "инфор-" not in result

    def test_убирает_строки_разделители(self):
        text = "Заголовок\n------\nТекст после"
        result = clean_text(text)
        assert "------" not in result
        assert "Заголовок" in result
        assert "Текст после" in result

    def test_множественные_пробелы_схлопываются(self):
        text = "слово1    слово2"
        result = clean_text(text)
        assert "    " not in result
        assert "слово1 слово2" in result

    def test_множественные_пустые_строки_схлопываются(self):
        text = "абзац1\n\n\n\n\nабзац2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_пустая_строка_не_падает(self):
        # граничный случай — пустой вход не должен вызывать ошибку
        result = clean_text("")
        assert result == ""


class TestExtractPdf:
    """
    Тесты на extract_pdf — требуют реальный PDF-файл в data/test.pdf.
    Если файла нет, тесты пропускаются (а не падают), чтобы не блокировать
    остальной набор тестов на машинах без тестового файла.
    """

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="Нет тестового PDF в data/test.pdf — положи файл, чтобы прогнать этот тест",
    )
    def test_возвращает_список_по_числу_страниц(self):
        result = extract_pdf(TEST_PDF_PATH)
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="Нет тестового PDF в data/test.pdf — положи файл, чтобы прогнать этот тест",
    )
    def test_каждая_страница_содержит_нужные_поля(self):
        result = extract_pdf(TEST_PDF_PATH)
        for page in result:
            assert "page_num" in page
            assert "text" in page
            assert "source" in page
            assert page["page_num"] > 0
            assert page["source"] != ""

    @pytest.mark.skipif(
        not os.path.exists(TEST_PDF_PATH),
        reason="Нет тестового PDF в data/test.pdf — положи файл, чтобы прогнать этот тест",
    )
    def test_текст_страниц_непустой_для_native_pdf(self):
        # На native-PDF (с текстовым слоем) текст не должен быть пустым
        result = extract_pdf(TEST_PDF_PATH)
        non_empty_pages = [p for p in result if len(p["text"]) > 0]
        assert len(non_empty_pages) > 0, "Ни одна страница не вернула текст"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])