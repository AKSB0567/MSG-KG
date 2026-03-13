"""
SEC Extractor: 10-K Data Collection
===================================
Automated scraper for SEC EDGAR filings. Specifically targets "Item 1 (Business)"
from 10-K filings and performs semantic chunking for the RAG pipeline.
"""

import os, re, json, time, pathlib, textwrap
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Company filing metadata ──────────────────────────────────────────────
FILINGS = [
    {"ticker": "AAPL", "cik": "0000320193", "accession": "000032019324000123", "filename": "aapl-20240928.htm"},
    {"ticker": "AMD",  "cik": "0000002488", "accession": "000000248825000012", "filename": "amd-20241228.htm"},
    {"ticker": "TGT",  "cik": "0000027419", "accession": "000002741925000018", "filename": "tgt-20250201.htm"},
    {"ticker": "WMT",  "cik": "0000104169", "accession": "000010416925000021", "filename": "wmt-20250131.htm"},
    {"ticker": "TSN",  "cik": "0000100493", "accession": "000010049325000095", "filename": "tsn-20250927.htm"},
    {"ticker": "MSFT", "cik": "0000789019", "accession": "000095017025100235", "filename": "msft-20250630.htm"},
]

BASE_DIR   = pathlib.Path(__file__).parent
ITEM1_DIR  = BASE_DIR / "data" / "item1"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

HEADERS = {
    "User-Agent": "MSGKG-Research/1.0 (Finance Research; contact@university.edu)",
    "Accept-Encoding": "gzip, deflate",
}

CHUNK_SIZE    = 512   # tokens (approximate by words)
CHUNK_OVERLAP = 64


def build_url(filing: dict) -> str:
    """Build the direct EDGAR document URL."""
    cik_clean = filing["cik"].lstrip("0")
    acc_no_dash = filing["accession"].replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_clean}/{acc_no_dash}/{filing['filename']}"
    )


def download_filing(url: str) -> str:
    """Download filing HTML from SEC EDGAR with rate limiting."""
    print(f"  Downloading: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(0.2)  # SEC rate limit: 10 req/sec max
    return resp.text


def extract_item1(html: str, ticker: str) -> str:
    """
    Extract Item 1 (Business) section from 10-K HTML.
    Uses multiple heuristic patterns to handle different filing formats.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove scripts, styles, hidden elements
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n")
    lines = [l.strip() for l in full_text.splitlines()]
    text = "\n".join(l for l in lines if l)

    # ── Strategy: find Item 1 start and Item 1A / Item 2 end ─────────
    # Common patterns in 10-K filings
    item1_start_patterns = [
        r'(?i)(?:^|\n)\s*(?:ITEM\s*1\.?\s*[\-—–]?\s*BUSINESS)',
        r'(?i)(?:^|\n)\s*(?:Item\s+1\.\s+Business)',
        r'(?i)(?:^|\n)\s*(?:PART\s+I\s*\n+\s*Item\s*1)',
    ]

    item1_end_patterns = [
        r'(?i)(?:^|\n)\s*(?:ITEM\s*1A\.?\s*[\-—–]?\s*RISK\s*FACTORS)',
        r'(?i)(?:^|\n)\s*(?:Item\s+1A\.\s+Risk\s+Factors)',
        r'(?i)(?:^|\n)\s*(?:ITEM\s*2\.?\s*[\-—–]?\s*PROPERTIES)',
        r'(?i)(?:^|\n)\s*(?:Item\s+2\.\s+Properties)',
    ]

    start_pos = None
    for pat in item1_start_patterns:
        m = re.search(pat, text)
        if m:
            start_pos = m.start()
            break

    end_pos = None
    if start_pos is not None:
        search_after = start_pos + 100  # skip past the header itself
        for pat in item1_end_patterns:
            m = re.search(pat, text[search_after:])
            if m:
                end_pos = search_after + m.start()
                break

    if start_pos is not None:
        item1_text = text[start_pos : end_pos] if end_pos else text[start_pos : start_pos + 100000]
    else:
        # Fallback: take first 50K characters of body
        print(f"  ⚠ Could not locate Item 1 header for {ticker}, using fallback extraction")
        item1_text = text[:50000]

    # Clean up
    item1_text = re.sub(r'\n{3,}', '\n\n', item1_text)
    item1_text = re.sub(r'Table of Contents', '', item1_text, flags=re.IGNORECASE)
    item1_text = item1_text.strip()

    return item1_text


def chunk_text(text: str, ticker: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks = []
    idx = 0
    page_estimate = 1

    while idx < len(words):
        end = min(idx + chunk_size, len(words))
        chunk_words = words[idx:end]
        chunk_text_str = " ".join(chunk_words)

        # Rough page estimate (250 words per page)
        page_estimate = (idx // 250) + 1

        chunks.append({
            "chunk_id": f"{ticker}_chunk_{len(chunks):04d}",
            "ticker": ticker,
            "section": "Item 1",
            "page_estimate": page_estimate,
            "start_word": idx,
            "end_word": end,
            "text": chunk_text_str,
        })

        idx += chunk_size - overlap
        if end >= len(words):
            break

    return chunks


def process_all():
    """Main pipeline: download, extract, chunk all filings."""
    ITEM1_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for filing in FILINGS:
        ticker = filing["ticker"]
        print(f"\n{'='*60}")
        print(f"Processing {ticker} ...")
        print(f"{'='*60}")

        txt_path = ITEM1_DIR / f"{ticker}_item1.txt"
        chunk_path = CHUNKS_DIR / f"{ticker}_chunks.json"

        # ── Download ─────────────────────────────────────────────
        try:
            url = build_url(filing)
            html = download_filing(url)
        except Exception as e:
            print(f"  ✗ Download failed for {ticker}: {e}")
            # Try alternate URL format
            try:
                alt_url = url.replace(filing["cik"].lstrip("0"), filing["cik"])
                html = download_filing(alt_url)
            except Exception as e2:
                print(f"  ✗ Alternate URL also failed: {e2}")
                summary.append({"ticker": ticker, "status": "FAILED", "error": str(e)})
                continue

        # ── Extract Item 1 ───────────────────────────────────────
        item1_text = extract_item1(html, ticker)
        txt_path.write_text(item1_text, encoding="utf-8")
        print(f"  ✓ Saved {txt_path.name} ({len(item1_text):,} chars)")

        # ── Chunk ────────────────────────────────────────────────
        chunks = chunk_text(item1_text, ticker)
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        print(f"  ✓ Saved {chunk_path.name} ({len(chunks)} chunks)")

        summary.append({
            "ticker": ticker,
            "status": "OK",
            "chars": len(item1_text),
            "chunks": len(chunks),
            "file": str(txt_path),
        })

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*60}")
    for s in summary:
        if s["status"] == "OK":
            print(f"  {s['ticker']:5s}  ✓  {s['chars']:>8,} chars  {s['chunks']:>4} chunks")
        else:
            print(f"  {s['ticker']:5s}  ✗  {s.get('error','unknown')}")

    return summary


if __name__ == "__main__":
    process_all()
