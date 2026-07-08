import re
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image


def classify_page(page) -> str:
    text = page.get_text().strip()
    if len(text) < 10:
        return "scan"
    return "native"


def ocr_page(page) -> str:
    pix = page.get_pixmap(dpi=300)
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image, lang="rus+eng")
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"^[\-_.\s]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: str) -> list[dict]:

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