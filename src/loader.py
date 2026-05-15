"""
PDF loader with citation-aware chunking.
Splits on paper structure (sections, paragraphs) instead of raw character counts.
"""
import os
import re
import fitz  # pymupdf


def load_pdf_text(file_path: str) -> str:
    """Extract full text from a PDF with page markers. Returns empty string on failure."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for i, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n[PAGE_{i}]\n{page_text}"
        doc.close()
        return text
    except Exception as e:
        print(f"  WARN: Failed to parse {file_path}: {e}")
        return ""


def _extract_year(meta: dict) -> str:
    """Extract year from PDF metadata (various date formats)."""
    import re
    for key in ("creationDate", "modDate", "date"):
        val = meta.get(key, "")
        if val:
            # D:20230101... or 2023-01-01 or just 2023
            m = re.search(r"(?:D:)?(\d{4})", str(val))
            if m:
                return m.group(1)
    return ""


def load_pdf_meta(file_path: str) -> dict:
    """
    Extract metadata (title, author, year) from a PDF.
    Falls back to filename if metadata is missing.
    """
    try:
        doc = fitz.open(file_path)
        meta = doc.metadata
        doc.close()

        # Clean empty/placeholder values
        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        year = _extract_year(meta)

        return {"title": title, "author": author, "year": year}
    except Exception:
        return {"title": "", "author": "", "year": ""}


def load_pdfs_from_dir(dir_path: str) -> list[dict]:
    """Load all PDFs from a directory. Returns [{'name': ..., 'text': ...}]."""
    papers = []
    for fname in os.listdir(dir_path):
        if fname.endswith(".pdf"):
            full_path = os.path.join(dir_path, fname)
            text = load_pdf_text(full_path)
            if text.strip():
                papers.append({"name": fname, "text": text})
    return papers


# --- Citation-aware chunking ---

_PAGE_RE = re.compile(r"\[PAGE_(\d+)\]")


def _extract_pages(text: str) -> str:
    """Extract page numbers from [PAGE_N] markers. Returns comma-separated string."""
    pages = _PAGE_RE.findall(text)
    return ",".join(pages) if pages else "?"


def _strip_page_markers(text: str) -> str:
    """Remove [PAGE_N] markers from text."""
    return _PAGE_RE.sub("", text).strip()

SECTION_PATTERNS = re.compile(
    r"\n\s*(?:\d+[\.\)]\s*)?"
    r"(?:Abstract|Introduction|Related\s*Work|Background|"
    r"Method|Methodology|Data|Empirical\s*Strategy|Results?|"
    r"Discussion|Conclusion|References?|Bibliography|Appendix|"
    r"Motivation|Model|Experiment|Evaluation|Findings|Policy\s*Implications)",
    re.IGNORECASE,
)


def split_on_sections(text: str) -> list[str]:
    """
    Split text on common academic section headers.
    This preserves the logical structure of a paper.
    Sections that are too long are further split by paragraphs.
    """
    # Find section boundaries
    matches = list(SECTION_PATTERNS.finditer(text))
    if len(matches) < 2:
        # No clear sections found, fall back to paragraph split
        return _split_paragraphs(text)

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) > 100:
            sections.append(section_text)

    return sections


def _split_paragraphs(text: str, max_chars: int = 1200) -> list[str]:
    """Split text by paragraph boundaries, merging short ones."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current) + len(p) < max_chars:
            current += ("\n\n" if current else "") + p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def chunk_papers(papers: list[dict]) -> list[dict]:
    """
    Citation-aware chunking pipeline:
    1. Split each paper into sections
    2. Split oversized sections into paragraphs

    Returns: [{'paper_name': ..., 'chunk_id': ..., 'text': ..., 'page': ...}]
    """
    chunks = []
    for paper in papers:
        sections = split_on_sections(paper["text"])

        # If section-level split didn't work well, fall back
        if not sections or len(sections) == 1:
            sections = _split_paragraphs(paper["text"])

        for i, section in enumerate(sections):
            if len(section) > 2000:
                # Further split long sections
                sub = _split_paragraphs(section, max_chars=1000)
                for j, sub_text in enumerate(sub):
                    if sub_text.strip() and len(sub_text) > 50:
                        chunks.append({
                            "paper_name": paper["name"],
                            "chunk_id": len(chunks),
                            "text": _strip_page_markers(sub_text.strip()),
                            "page": _extract_pages(sub_text.strip()),
                        })
            else:
                if section.strip() and len(section) > 50:
                    chunks.append({
                        "paper_name": paper["name"],
                        "chunk_id": len(chunks),
                        "text": _strip_page_markers(section.strip()),
                        "page": _extract_pages(section.strip()),
                    })
    return chunks
