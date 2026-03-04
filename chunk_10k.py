"""
Chunker: Section-Aware Recursive Chunking for 10-K Filings
===========================================================
Extracts Item 1 (Business) from cleaned 10-K SEC filings and chunks them
using a recursive section-aware strategy for better retrieval quality.

Strategy:
- Extract clean text body, stripping HTML/headers
- Locate Item 1 section
- Split on sentence boundaries (period + space after capital/number)
- Detect section headers within text and annotate chunks
- Merge sentences into ~512-word chunks with 80-word overlap
"""

import os
import re
import json
import pathlib

BASE_DIR   = pathlib.Path(__file__).parent
KG_DIR     = BASE_DIR / "MSGKG"
CHUNKS_DIR = BASE_DIR / "data" / "chunks"

# ── Mapping: accession → ticker ─────────────────────────────────────────
FILE_TO_TICKER = {
    "0000002488-25-000012": "AMD",
    "0000003499-25-000004": "ALX",
    "0000003570-25-000033": "LNG",
}

# ── Chunking parameters ────────────────────────────────────────────────
CHUNK_SIZE    = 512   # target words per chunk
CHUNK_OVERLAP = 80    # overlap words
MIN_CHUNK_LEN = 50    # min words to keep a chunk


# ── Section header patterns (within SEC Item 1 text) ───────────────────
SECTION_PATTERNS = [
    (r'(?:^|\s)Overview\b',                            "Overview"),
    (r'(?:^|\s)Our\s+Strategy\b',                     "Strategy"),
    (r'(?:^|\s)Strategy\b',                            "Strategy"),
    (r'(?:^|\s)Our\s+Business\b',                     "Business Segments"),
    (r'(?:^|\s)Business\s+Segments?\b',               "Business Segments"),
    (r'(?:^|\s)Products?\b',                           "Products"),
    (r'(?:^|\s)Competition\b',                         "Competition"),
    (r'(?:^|\s)Research\s+and\s+Development\b',       "R&D"),
    (r'(?:^|\s)Manufacturing\b',                       "Manufacturing"),
    (r'(?:^|\s)Human\s+Capital\b',                     "Human Capital"),
    (r'(?:^|\s)Mission,?\s+Culture\b',                "Mission & Culture"),
    (r'(?:^|\s)Sales\s+and\s+Marketing\b',            "Sales & Marketing"),
    (r'(?:^|\s)Intellectual\s+Property\b',            "IP & Licensing"),
    (r'(?:^|\s)Customers?\b',                          "Customers"),
    (r'(?:^|\s)Government\s+Regulations?\b',          "Regulations"),
    (r'(?:^|\s)Environmental\s+Regulations?\b',       "Environmental"),
    (r'(?:^|\s)Diversity,?\s+Belonging\b',            "DB&I"),
    (r'(?:^|\s)Total\s+Rewards?\b',                   "Total Rewards"),
    (r'(?:^|\s)Development\b',                         "Development"),
    (r'(?:^|\s)Employee\s+Voice\b',                   "Employee Voice"),
    (r'(?:^|\s)Additional\s+Information\b',           "Additional Info"),
    (r'(?:^|\s)Seasonality\b',                         "Seasonality"),
    (r'(?:^|\s)Backlog\b',                             "Backlog"),
    (r'(?:^|\s)Data\s+Center\s+Segment\b',           "Data Center Segment"),
    (r'(?:^|\s)Client\s+Segment\b',                   "Client Segment"),
    (r'(?:^|\s)Gaming\s+Segment\b',                   "Gaming Segment"),
    (r'(?:^|\s)Embedded\s+Segment\b',                 "Embedded Segment"),
]


def extract_text_body(raw: str) -> str:
    """Strip SEC header and HTML artifacts from raw filing text."""
    # Find content start
    markers = [
        r'ITEM\s+1\.?\s*[\-\u2013\u2014]?\s*BUSINESS',
        r'UNITED\s+STATES\s+SECURITIES',
        r'10-K\s+1\s+\w',
    ]
    start_pos = 0
    for marker in markers:
        m = re.search(marker, raw, re.IGNORECASE)
        if m:
            start_pos = m.start()
            break

    text = raw[start_pos:]

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    # Remove [TABLE REMOVED] artifacts
    text = re.sub(r'\[TABLE REMOVED\]', ' ', text)
    # Remove "Table of Contents" page markers
    text = re.sub(r'Table of Contents', ' ', text, flags=re.IGNORECASE)
    # Remove page numbers like "1 " "12 " at line beginnings
    text = re.sub(r'(?:^|\n)\s*\d{1,3}\s+(?=[A-Z])', ' ', text)
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    return text.strip()


def extract_item1(text: str, ticker: str) -> str:
    """
    Extract Item 1 (Business) section from cleaned text.
    
    Key challenge: 10-K filings have a Table of Contents early on that
    mentions "ITEM 1A. Risk Factors" right after "ITEM 1. BUSINESS".
    We must skip past TOC entries to find the actual section headers.
    
    Strategy: Find ALL matches of "ITEM 1. BUSINESS" and use the LAST one
    (the actual section header). Then find "ITEM 1A" at least 5000 chars
    after that to skip past any short TOC references.
    """
    # Find ALL matches of Item 1 header — use last one (actual header, not TOC)
    item1_pat = r'ITEM\s+1\.?\s*[\-\u2013\u2014]?\s*BUSINESS'
    matches = list(re.finditer(item1_pat, text, re.IGNORECASE))

    if not matches:
        print(f"  [WARN] Could not find Item 1 for {ticker}, using first 80K chars")
        return text[:80000]

    # Use the LAST match — that's the actual section header, not the TOC entry
    start_pos = matches[-1].start()

    # Find Item 1A (Risk Factors) — must be at least 5000 chars after Item 1
    # to skip any remaining TOC references
    item1a_pat = r'ITEM\s+1A\.?\s*[\-\u2013\u2014]?\s*RISK\s+FACTORS'
    min_gap = 5000  # Item 1 Business section is always > 5000 chars
    search_from = start_pos + min_gap
    
    end_pos = None
    for m in re.finditer(item1a_pat, text, re.IGNORECASE):
        if m.start() > search_from:
            end_pos = m.start()
            break

    if end_pos is None:
        end_pos = min(start_pos + 120000, len(text))

    return text[start_pos:end_pos].strip()


def split_sentences(text: str) -> list:
    """
    Split text into sentences.
    Handles abbreviations, numbers, and SEC-specific patterns.
    """
    # Replace common abbreviations that would cause false splits
    text_clean = text
    abbrevs = [
        r'(?:Inc|Corp|Ltd|Co|Jr|Sr|Dr|Mr|Mrs|Ms|Prof|Rev|'
        r'Dept|Univ|Est|Assn|Bros|etc|vs|e\.g|i\.e|'
        r'approx|No|Vol|Dept|Ph\.D|U\.S|D\.C)\.'
    ]
    for abbr in abbrevs:
        text_clean = re.sub(abbr, lambda m: m.group().replace('.', '<DOT>'), text_clean)

    # Split on sentence boundaries: period/question/exclamation followed by space and capital
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text_clean)

    # Restore abbreviated dots
    sentences = [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]
    return sentences


def detect_section(text: str) -> str:
    """Detect which section a text fragment belongs to."""
    first_80 = text[:80]
    for pattern, section_name in SECTION_PATTERNS:
        if re.search(pattern, first_80, re.IGNORECASE):
            return section_name
    return ""


def recursive_chunk(text: str, ticker: str,
                    chunk_size: int = CHUNK_SIZE,
                    overlap: int = CHUNK_OVERLAP) -> list:
    """
    Sentence-aware chunking with section detection:
    1. Split text into sentences
    2. Group sentences into chunks of ~chunk_size words
    3. Detect section boundaries and annotate
    4. Add overlap for context continuity
    """
    sentences = split_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_words = []
    current_section = "Item 1"
    word_offset = 0

    def flush_chunk():
        nonlocal word_offset
        if len(current_words) < MIN_CHUNK_LEN:
            return

        chunk_text = " ".join(current_words)
        word_count = len(current_words)

        chunks.append({
            "chunk_id": f"{ticker}_chunk_{len(chunks):04d}",
            "ticker": ticker,
            "section": current_section,
            "page_estimate": word_offset // 250 + 1,
            "start_word": max(0, word_offset - overlap),
            "end_word": word_offset + word_count,
            "text": chunk_text,
        })

    for sentence in sentences:
        sent_words = sentence.split()

        # Check for section transition
        new_section = detect_section(sentence)
        if new_section:
            current_section = new_section

        # If adding sentence would exceed chunk_size, flush first
        if len(current_words) + len(sent_words) > chunk_size and current_words:
            flush_chunk()
            word_offset += len(current_words)
            # Keep overlap
            overlap_words = current_words[-overlap:] if len(current_words) > overlap else current_words[:]
            current_words = overlap_words[:]

        current_words.extend(sent_words)

    # Flush remaining
    if current_words:
        flush_chunk()

    return chunks


def process_file(filepath: pathlib.Path, ticker: str) -> list:
    """Process a single cleaned 10-K file to chunks."""
    print(f"\nProcessing {ticker}: {filepath.name}")

    raw = filepath.read_text(encoding="utf-8", errors="replace")
    print(f"  Raw file: {len(raw):,} chars")

    # Extract clean text
    text = extract_text_body(raw)
    print(f"  Clean text: {len(text):,} chars")

    # Extract Item 1
    item1 = extract_item1(text, ticker)
    word_count = len(item1.split())
    print(f"  Item 1: {len(item1):,} chars ({word_count:,} words)")

    # Save Item 1 text
    item1_path = BASE_DIR / "data" / "item1" / f"{ticker}_item1.txt"
    item1_path.parent.mkdir(parents=True, exist_ok=True)
    item1_path.write_text(item1, encoding="utf-8")
    print(f"  Saved {item1_path.name}")

    # Chunk
    chunks = recursive_chunk(item1, ticker)
    total_words = sum(len(c["text"].split()) for c in chunks)
    avg_words = total_words // max(len(chunks), 1)
    print(f"  => {len(chunks)} chunks (avg {avg_words} words/chunk, total {total_words:,} words)")

    # Section distribution
    sections = {}
    for c in chunks:
        sec = c["section"]
        sections[sec] = sections.get(sec, 0) + 1
    for sec, cnt in sections.items():
        print(f"     [{sec}]: {cnt} chunks")

    return chunks


def process_all():
    """Process all cleaned 10-K files."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for filepath in sorted(KG_DIR.glob("cleaned_10-K_*.txt")):
        accession = filepath.stem.replace("cleaned_10-K_", "")
        ticker = FILE_TO_TICKER.get(accession)
        if ticker is None:
            print(f"  [WARN] Unknown accession: {accession}, skipping")
            continue

        chunks = process_file(filepath, ticker)

        chunk_path = CHUNKS_DIR / f"{ticker}_chunks.json"
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"  Saved {chunk_path.name}")

        summary.append({
            "ticker": ticker,
            "chunks": len(chunks),
            "total_words": sum(len(c["text"].split()) for c in chunks),
        })

    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"  {s['ticker']}: {s['chunks']} chunks, {s['total_words']:,} words")

    return summary


if __name__ == "__main__":
    process_all()
