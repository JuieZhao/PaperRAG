"""
PDF loader with citation-aware chunking and table extraction.
Splits on paper structure (sections, paragraphs) instead of raw character counts.
"""
from __future__ import annotations

import os
import re
import hashlib
import fitz  # pymupdf


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file (for deduplication)."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


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


def extract_tables_from_pdf(file_path: str) -> list[dict]:
    """
    Extract tables from a PDF and convert to Markdown format.

    Uses PyMuPDF's built-in table detection (fitz.Page.find_tables).
    Returns: [{'page': int, 'markdown': str, 'rows': int, 'cols': int}]

    Note: find_tables() requires PyMuPDF >= 1.23.0.
    Falls back gracefully on older versions.
    """
    tables = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc, 1):
            try:
                found = page.find_tables()
            except AttributeError:
                # find_tables not available (PyMuPDF < 1.23.0)
                doc.close()
                return tables

            for table in found:
                data = table.extract()
                if not data or len(data) < 2:
                    continue

                # Convert to Markdown table
                rows = len(data)
                cols = max(len(row) for row in data) if data else 0
                md_lines = []

                # Header row
                header = [str(cell or "").replace("\n", " ").strip() for cell in data[0]]
                # Pad header to match max cols
                while len(header) < cols:
                    header.append("")
                md_lines.append("| " + " | ".join(header) + " |")
                md_lines.append("| " + " | ".join(["---"] * cols) + " |")

                # Data rows
                for row in data[1:]:
                    cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
                    while len(cells) < cols:
                        cells.append("")
                    md_lines.append("| " + " | ".join(cells) + " |")

                tables.append({
                    "page": page_num,
                    "markdown": "\n".join(md_lines),
                    "rows": rows,
                    "cols": cols,
                })
        doc.close()
    except Exception as e:
        print(f"  WARN: Table extraction failed for {file_path}: {e}")

    return tables


def _extract_year(meta: dict) -> str:
    """Extract year from PDF metadata (various date formats)."""
    for key in ("creationDate", "modDate", "date"):
        val = meta.get(key, "")
        if val:
            m = re.search(r"(?:D:)?(\d{4})", str(val))
            if m:
                return m.group(1)
    return ""


def load_pdf_meta(file_path: str) -> dict:
    """
    Extract metadata (title, author, year, file_hash) from a PDF.
    Falls back to filename if metadata is missing.
    """
    try:
        doc = fitz.open(file_path)
        meta = doc.metadata
        doc.close()

        title = (meta.get("title") or "").strip()
        author = (meta.get("author") or "").strip()
        year = _extract_year(meta)
        file_hash = compute_file_hash(file_path)

        return {
            "title": title,
            "author": author,
            "year": year,
            "file_hash": file_hash,
        }
    except Exception:
        return {"title": "", "author": "", "year": "", "file_hash": ""}


def load_pdfs_from_dir(dir_path: str) -> list[dict]:
    """Load all PDFs from a directory. Returns [{'name': ..., 'text': ...}]."""
    papers = []
    for fname in os.listdir(dir_path):
        if fname.endswith(".pdf"):
            full_path = os.path.join(dir_path, fname)
            text = load_pdf_text(full_path)
            if text.strip():
                papers.append({"name": fname, "text": text, "path": full_path})
    return papers


# ── Citation-aware chunking ──────────────────────────────────

_PAGE_RE = re.compile(r"\[PAGE_(\d+)\]")


def _extract_pages(text: str) -> str:
    """Extract page numbers from [PAGE_N] markers."""
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
    Sections that are too long are further split by paragraphs.
    """
    matches = list(SECTION_PATTERNS.finditer(text))
    if len(matches) < 2:
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


def chunk_papers(
    papers: list[dict],
    extract_tables: bool = True,
) -> list[dict]:
    """
    Citation-aware chunking pipeline with optional table extraction:
    1. Split each paper into sections
    2. Split oversized sections into paragraphs
    3. Optionally extract tables and append as separate chunks

    papers: [{'name': ..., 'text': ..., 'path'?: ...}]
    Returns: [{'paper_name': ..., 'chunk_id': ..., 'text': ..., 'page': ...,
               'is_table': bool}]
    """
    chunks = []
    chunk_id = 0

    for paper in papers:
        sections = split_on_sections(paper["text"])

        if not sections or len(sections) == 1:
            sections = _split_paragraphs(paper["text"])

        for section in sections:
            if len(section) > 2000:
                sub = _split_paragraphs(section, max_chars=1000)
                for sub_text in sub:
                    if sub_text.strip() and len(sub_text) > 50:
                        chunks.append({
                            "paper_name": paper["name"],
                            "chunk_id": chunk_id,
                            "text": _strip_page_markers(sub_text.strip()),
                            "page": _extract_pages(sub_text.strip()),
                            "is_table": False,
                        })
                        chunk_id += 1
            else:
                if section.strip() and len(section) > 50:
                    chunks.append({
                        "paper_name": paper["name"],
                        "chunk_id": chunk_id,
                        "text": _strip_page_markers(section.strip()),
                        "page": _extract_pages(section.strip()),
                        "is_table": False,
                    })
                    chunk_id += 1

        # ── Table extraction ──
        if extract_tables and "path" in paper:
            tables = extract_tables_from_pdf(paper["path"])
            for t in tables:
                table_text = (
                    f"[TABLE from page {t['page']} — {t['rows']}×{t['cols']}]\n"
                    f"{t['markdown']}"
                )
                chunks.append({
                    "paper_name": paper["name"],
                    "chunk_id": chunk_id,
                    "text": table_text,
                    "page": str(t["page"]),
                    "is_table": True,
                })
                chunk_id += 1

    return chunks
