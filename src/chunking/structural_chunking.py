import pdfplumber
from typing import Optional


def detect_sections_by_font_size(pdf_path: str, min_header_font_size: float = 12.0) -> list[dict]:

    sections = []
    
    with pdfplumber.open(pdf_path) as pdf:
        current_section = None
        current_text_parts = []
        
        for page_num, page in enumerate(pdf.pages):
            # Get characters with font information
            chars = page.chars
            
            if not chars:
                continue
            
            # Group characters by line (approximate)
            current_y = None
            line_chars = []
            lines = []
            
            for char in chars:
                if current_y is None or abs(char["top"] - current_y) > 5:
                    # New line
                    if line_chars:
                        lines.append(line_chars)
                    line_chars = [char]
                    current_y = char["top"]
                else:
                    line_chars.append(char)
            
            if line_chars:
                lines.append(line_chars)
            
            # Analyze each line
            for line in lines:
                if not line:
                    continue
                
                # Get average font size for line
                font_sizes = [char.get("size", 12) for char in line]
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                
                # Get line text
                line_text = "".join(char.get("text", "") for char in line).strip()
                
                # Check if this is a header
                if avg_font_size >= min_header_font_size and line_text:
                    # Save previous section
                    if current_section:
                        current_section["text"] = "\n".join(current_text_parts)
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = {
                        "title": line_text,
                        "start_page": page_num,
                        "start_y": current_y,
                        "text": "",
                    }
                    current_text_parts = [line_text]
                else:
                    # Regular text - add to current section
                    if line_text:
                        current_text_parts.append(line_text)
        
        # Save last section
        if current_section:
            current_section["text"] = "\n".join(current_text_parts)
            sections.append(current_section)
    
    return sections


def split_by_sections(pages: list[dict], sections: list[dict]) -> list[dict]:

    if not sections:
        # No sections detected - return single section with all text
        all_text = "\n".join(page["text"] for page in pages)
        return [{
            "title": "Document",
            "start_page": 0,
            "text": all_text,
            "source": pages[0]["source"] if pages else "unknown",
        }]

    result = []
    for section in sections:
        section_text = section.get("text", "")
        if section_text.strip():
            result.append({
                "title": section.get("title", "Untitled"),
                "start_page": section.get("start_page", 0),
                "text": section_text,
                "source": pages[0]["source"] if pages else "unknown",
            })
    
    return result if result else [{
        "title": "Document",
        "start_page": 0,
        "text": "\n".join(page["text"] for page in pages),
        "source": pages[0]["source"] if pages else "unknown",
    }]


def chunk_with_structure(pages: list[dict], pdf_path: Optional[str] = None) -> list[dict]:

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # Detect sections if PDF path provided
    sections = []
    if pdf_path:
        try:
            sections = detect_sections_by_font_size(pdf_path)
        except Exception as e:
            print(f"Warning: structural detection failed ({e}), falling back to regular chunking")
    
    # If no sections detected, use all text as single section
    if not sections:
        all_text = "\n".join(page["text"] for page in pages)
        sections = [{
            "title": "Document",
            "start_page": 0,
            "text": all_text,
            "source": pages[0]["source"] if pages else "unknown",
        }]
    else:
        # Map sections to pages
        sections = split_by_sections(pages, sections)
    
    # Chunk each section independently
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
    )
    
    chunks = []
    chunk_idx = 0
    
    for section in sections:
        section_text = section["text"]
        if not section_text.strip():
            continue

        section_chunks = splitter.split_text(section_text)
        
        for chunk_text in section_chunks:
            chunks.append({
                "text": chunk_text,
                "source": section.get("source", "unknown"),
                "page_num": section.get("start_page", 0),
                "chunk_idx": chunk_idx,
                "section_title": section.get("title", "Untitled"),
            })
            chunk_idx += 1
    
    return chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../pipeline")
    from src.pipeline.extraction import extract_pdf
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.pdf"
    
    print("=== Detecting sections ===")
    sections = detect_sections_by_font_size(pdf_path)
    print(f"Found {len(sections)} sections:")
    for section in sections:
        print(f"  - {section['title']} (page {section['start_page']})")
    print()
    
    print("=== Chunking with structure ===")
    pages = extract_pdf(pdf_path)
    chunks = chunk_with_structure(pages, pdf_path)
    print(f"Created {len(chunks)} chunks")
    if chunks:
        print(f"First chunk section: {chunks[0].get('section_title', 'N/A')}")
        print(f"First chunk text: {chunks[0]['text'][:150]}")
