"""
Извлечение текста из PDF.
Определяет для каждой страницы, есть ли текстовый слой (native)
или это скан (нужен OCR), и возвращает чистый текст.
"""

import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# На Windows pytesseract не всегда находит tesseract.exe сам.
# Если запуск падает с ошибкой "tesseract is not installed",
# раскомментируй строку ниже и поправь путь под свою установку:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def classify_page(page) -> str:
    """Определяет тип страницы: 'native' (есть текстовый слой) или 'scan' (нужен OCR)."""
    text = page.get_text().strip()
    if len(text) < 10:
        return "scan"
    return "native"


def ocr_page(page) -> str:
    """Растеризует страницу и распознаёт текст через Tesseract."""
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image, lang="rus+eng")
    return text


def clean_text(text: str) -> str:
    """Чистит текст: убирает двойные пробелы, склеивает переносы слов,
    удаляет строки-разделители (из тире/точек)."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^[\-_.\s]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: str) -> list[dict]:
    """
    Открывает PDF и возвращает список словарей по страницам:
    {page_num, text, source}
    """
    doc = fitz.open(path)
    source = path.replace("\\", "/").split("/")[-1]
    pages = []

    for i, page in enumerate(doc):
        page_type = classify_page(page)
        raw_text = page.get_text() if page_type == "native" else ocr_page(page)

        pages.append({
            "page_num": i + 1,
            "text": clean_text(raw_text),
            "source": source,
        })

    doc.close()
    return pages


if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"
    result = extract_pdf(test_path)
    print(f"Страниц обработано: {len(result)}")
    for p in result[:2]:
        print(f"--- стр. {p['page_num']} ({p['source']}) ---")
        print(p["text"][:300])
        print()