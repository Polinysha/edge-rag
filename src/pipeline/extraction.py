"""
PDF text extraction.
Determines for each page whether it has a text layer (native)
or is a scan (needs OCR), and returns clean text.
"""

import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# On Windows, pytesseract doesn't always find tesseract.exe automatically.
# If extraction fails with "tesseract is not installed",
# uncomment the line below and fix the path for your installation:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def classify_page(page) -> str:
    """Determines page type: 'native' (has a text layer) or 'scan' (needs OCR)."""
    text = page.get_text().strip()
    if len(text) < 10:
        return "scan"
    return "native"


def ocr_page(page) -> str:
    """Rasterizes the page and recognizes text via Tesseract."""
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image, lang="rus+eng")
    return text


def clean_text(text: str) -> str:
    """Cleans text: removes double spaces, joins hyphenated line breaks,
    removes separator lines (made of dashes/dots)."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^[\-_.\s]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: str) -> list[dict]:
    """
    Opens a PDF and returns a list of dicts per page:
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
    print(f"Pages processed: {len(result)}")
    for p in result[:2]:
        print(f"--- page {p['page_num']} ({p['source']}) ---")
        print(p["text"][:300])
        print()