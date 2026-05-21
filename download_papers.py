"""
Paper downloader with arXiv API + RSS support.

Sources:
  1. arXiv API search by keyword/category
  2. arXiv RSS feeds (new papers in category)
  3. Manual URL list (backward compatible)
  4. Proxy support for users behind firewalls
"""
from __future__ import annotations

import os
import sys
import time
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

# Read proxy from env var (e.g. HTTP_PROXY=http://127.0.0.1:7890)
_proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
PROXY = {"http": _proxy_url, "https": _proxy_url} if _proxy_url else None

# Timeout for requests (seconds)
TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "120"))


# ═══════════════════════════════════════════
#  arXiv API
# ═══════════════════════════════════════════

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_RSS = "http://rss.arxiv.org/rss"


def search_arxiv(
    query: str,
    max_results: int = 10,
    categories: str | None = None,
    days_back: int = 30,
) -> list[dict]:
    """
    Search arXiv by keyword.

    Returns list of {title, arxiv_id, pdf_url, authors, published, abstract}.
    """
    search_query = f"all:{query}"
    if categories:
        search_query = f"({search_query}) AND cat:{categories}"

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(ARXIV_API, params=params, timeout=TIMEOUT, proxies=PROXY)
        resp.raise_for_status()
    except Exception as e:
        print(f"  arXiv API error: {e}")
        return []

    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        arxiv_id = entry.id.split("/abs/")[-1]
        # Remove version suffix for PDF URL
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        papers.append({
            "title": entry.title.strip().replace("\n", " "),
            "arxiv_id": clean_id,
            "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
            "authors": ", ".join(a.get("name", "") for a in entry.get("authors", [])),
            "published": entry.published,
            "abstract": entry.summary.strip()[:300] if hasattr(entry, "summary") else "",
        })

    return papers


def search_arxiv_rss(
    category: str = "econ.GN",
    max_results: int = 10,
) -> list[dict]:
    """
    Fetch latest papers from arXiv RSS feed (new today/yesterday).

    Common categories: econ.GN, cs.AI, stat.ML, q-fin.EC, physics.soc-ph
    """
    url = f"{ARXIV_RSS}/{category}/new"
    try:
        resp = requests.get(url, timeout=TIMEOUT, proxies=PROXY)
        resp.raise_for_status()
    except Exception as e:
        print(f"  arXiv RSS error: {e}")
        return []

    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries[:max_results]:
        arxiv_id = entry.id.split("/abs/")[-1]
        clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
        papers.append({
            "title": entry.title.strip().replace("\n", " "),
            "arxiv_id": clean_id,
            "pdf_url": f"https://arxiv.org/pdf/{clean_id}.pdf",
            "authors": entry.get("author", ""),
            "published": entry.get("published", ""),
            "abstract": entry.get("summary", "")[:300],
        })

    return papers


# ═══════════════════════════════════════════
#  Manual URL list (backward compatible)
# ═══════════════════════════════════════════

PAPERS = [
    {
        "name": "OECD_GVC_Development_Report_2023.pdf",
        "url": "https://www.wto.org/english/res_e/booksp_e/gvc_dev_rep_2023_e.pdf",
        "title": "Global Value Chain Development Report 2023 (WTO/ADB/OECD)",
    },
    {
        "name": "NBER_w26580_GVC_Gravity.pdf",
        "url": "https://www.nber.org/system/files/working_papers/w26580/w26580.pdf",
        "title": "Global Value Chains and Trade Elasticities (Antras et al., NBER 26580)",
    },
    {
        "name": "LMDI_IO_Energy_Emissions_China_2022.pdf",
        "url": "https://link.springer.com/content/pdf/10.1186/s40008-022-00270-w.pdf",
        "title": "LMDI Input-Output analysis of energy-related CO2 emissions in China",
    },
    {
        "name": "IMF_GVC_Bottleneck_Imports_2023.pdf",
        "url": "https://www.imf.org/-/media/Files/Publications/WP/2023/English/wpiea2023050-print-pdf.ashx",
        "title": "Global Value Chain Disruptions and Inflation (IMF WP 2023)",
    },
    {
        "name": "WTO_World_Trade_Report_2023_Re-globalization.pdf",
        "url": "https://www.wto.org/english/res_e/booksp_e/wtr23_e/wtr23_e.pdf",
        "title": "WTO World Trade Report 2023: Re-globalization",
    },
]


# ═══════════════════════════════════════════
#  Download logic
# ═══════════════════════════════════════════

def download_paper(paper: dict) -> bool:
    """Download a single paper. Returns True on success."""
    filepath = os.path.join(PAPERS_DIR, paper["name"])
    if os.path.exists(filepath):
        print(f"  SKIP (exists): {paper['name']}")
        return True
    try:
        resp = requests.get(paper["url"], timeout=TIMEOUT, proxies=PROXY)
        if resp.status_code == 200:
            # Basic PDF validation
            content = resp.content
            if not content.startswith(b"%PDF"):
                print(f"  WARN: Not a valid PDF: {paper['name']}")
                return False
            with open(filepath, "wb") as f:
                f.write(content)
            kb = os.path.getsize(filepath) / 1024
            print(f"  OK: {paper['name']} ({kb:.0f} KB)")
            return True
        else:
            print(f"  FAIL HTTP {resp.status_code}: {paper['name']}")
            return False
    except Exception as e:
        print(f"  FAIL: {paper['name']} - {str(e)[:80]}")
        return False


def download_from_arxiv(paper: dict) -> bool:
    """Download an arXiv paper, naming it by arxiv ID."""
    name = f"arXiv_{paper['arxiv_id']}.pdf"
    filepath = os.path.join(PAPERS_DIR, name)
    if os.path.exists(filepath):
        print(f"  SKIP (exists): {name}")
        return True
    try:
        resp = requests.get(paper["pdf_url"], timeout=TIMEOUT, proxies=PROXY)
        if resp.status_code == 200:
            content = resp.content
            if not content.startswith(b"%PDF"):
                print(f"  WARN: arXiv returned non-PDF: {name}")
                return False
            with open(filepath, "wb") as f:
                f.write(content)
            kb = os.path.getsize(filepath) / 1024
            print(f"  OK arXiv: {name} ({kb:.0f} KB) — {paper['title'][:60]}")
            return True
        else:
            print(f"  FAIL HTTP {resp.status_code}: {name}")
            return False
    except Exception as e:
        print(f"  FAIL: {name} - {str(e)[:80]}")
        return False


# ═══════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PaperRAG Downloader")
    sub = parser.add_subparsers(dest="command")

    # arxiv-search
    p = sub.add_parser("arxiv-search", help="Search arXiv by keyword")
    p.add_argument("query", help="Search query")
    p.add_argument("-n", "--max", type=int, default=5, help="Max results (default: 5)")
    p.add_argument("-c", "--category", help="arXiv category (e.g. econ.GN, cs.AI)")
    p.add_argument("--dry-run", action="store_true", help="Search only, no download")

    # arxiv-rss
    p = sub.add_parser("arxiv-rss", help="Fetch latest from arXiv RSS")
    p.add_argument("-c", "--category", default="econ.GN", help="Category (default: econ.GN)")
    p.add_argument("-n", "--max", type=int, default=5, help="Max results (default: 5)")
    p.add_argument("--dry-run", action="store_true", help="Fetch only, no download")

    # manual (backward compatible)
    p = sub.add_parser("manual", help="Download from built-in URL list")

    args = parser.parse_args()

    if args.command == "arxiv-search":
        print(f"arXiv search: \"{args.query}\" (max {args.max})")
        papers = search_arxiv(args.query, max_results=args.max, categories=args.category)
        print(f"Found: {len(papers)} papers\n")
        if args.dry_run:
            for p in papers:
                print(f"  [{p['arxiv_id']}] {p['title']}")
                print(f"    Authors: {p['authors']}")
                print()
        else:
            ok = sum(1 for p in papers if download_from_arxiv(p))
            print(f"\nDownloaded: {ok}/{len(papers)}")

    elif args.command == "arxiv-rss":
        print(f"arXiv RSS: {args.category} (max {args.max})")
        papers = search_arxiv_rss(category=args.category, max_results=args.max)
        print(f"Found: {len(papers)} papers\n")
        if args.dry_run:
            for p in papers:
                print(f"  [{p['arxiv_id']}] {p['title']}")
                print(f"    Authors: {p['authors']}")
                print()
        else:
            ok = sum(1 for p in papers if download_from_arxiv(p))
            print(f"\nDownloaded: {ok}/{len(papers)}")

    elif args.command == "manual":
        print(f"Downloading to: {PAPERS_DIR}\n")
        ok = 0
        for p in PAPERS:
            print(f"  {p['title'][:80]}...")
            if download_paper(p):
                ok += 1
        print(f"\nResult: {ok}/{len(PAPERS)} downloaded")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
