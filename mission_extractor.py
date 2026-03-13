"""
Mission Statement Extractor — Simple, Robust Pipeline
======================================================
Extracts mission statements from 91 SEC 10-K filings using:
  Stage 1: Strip SEC SGML headers + extract Item 1 section
  Stage 2: Sliding window + keyword scoring to find candidates
  Stage 3: Rule-based selection with tier classification
  Stage 4: Update companies_registry.json

Usage:  python mission_extractor.py
"""

import json, re, pathlib, time

BASE_DIR    = pathlib.Path(__file__).parent
SEC_DIR     = BASE_DIR / "MSGKG" / "data" / "data"
TTL_DIR     = BASE_DIR / "MSGKG" / "data" / "output"
REGISTRY    = BASE_DIR / "companies_registry.json"
OUTPUT_FILE = BASE_DIR / "companies_registry.json"

# ── Tier 1: Explicit mission language ────────────────────────────────────
TIER1_PATTERNS = [
    (r'(?:our|the\s+company\'?s?)\s+mission\s+(?:is\s+)?(?:to\s+)?(.{20,300}?)(?:\.|$)', 1.0),
    (r'(?:our|the\s+company\'?s?)\s+purpose\s+(?:is\s+)?(?:to\s+)?(.{20,300}?)(?:\.|$)', 0.95),
    (r'(?:our|the\s+company\'?s?)\s+vision\s+(?:is\s+)?(?:to\s+)?(.{20,300}?)(?:\.|$)', 0.90),
    (r'mission\s*:\s*(.{20,300}?)(?:\.|$)', 0.95),
    (r'mission\s+statement\s*[:\-–]\s*(.{20,300}?)(?:\.|$)', 1.0),
]

# ── Tier 2: Implied purpose language ────────────────────────────────────
TIER2_KEYWORDS = [
    ("we are committed to", 0.8),
    ("we are dedicated to", 0.8),
    ("we strive to", 0.75),
    ("we aim to", 0.75),
    ("we seek to", 0.7),
    ("our goal is", 0.7),
    ("we believe in", 0.65),
    ("we exist to", 0.9),
    ("our core purpose", 0.85),
    ("we aspire to", 0.7),
]

# ── Tier 3: Business description language ───────────────────────────────
TIER3_KEYWORDS = [
    ("is a leading", 0.5),
    ("is a global", 0.5),
    ("is one of the", 0.45),
    ("is a diversified", 0.5),
    ("is a premier", 0.5),
    ("provides a wide range", 0.4),
    ("we deliver", 0.45),
    ("we provide", 0.4),
    ("we offer", 0.4),
    ("focused on delivering", 0.5),
    ("is a provider of", 0.45),
    ("is engaged in", 0.4),
    ("principal business", 0.4),
]

# ── HCM section markers: deprioritize matches in these sections ─────────
HCM_MARKERS = [
    "human capital", "human resources", "workforce", "our people",
    "our employees", "our colleagues", "talent acquisition",
    "diversity and inclusion", "diversity, equity", "compensation and benefits",
    "health and safety", "workplace culture", "employee engagement",
    "training and development", "retention",
]

# ── Skip patterns: boilerplate that should never be a mission ───────────
SKIP_PATTERNS = [
    r'table of contents', r'securities and exchange', r'form 10-k',
    r'pursuant to section', r'commission file', r'registrant',
    r'incorporated by reference', r'annual report', r'fiscal year ended',
    r'check mark', r'transition report', r'indicate by', r'interactive data',
    r'edgar', r'accession', r'sgml', r'acceptance-datetime',
    r'conformed submission', r'irs number', r'zip code',
    r'telephone number', r'business address', r'mail address',
    r'par value per share', r'common stock', r'trading symbol',
    r'filed as of date', r'standard industrial',
]


def strip_sec_header(text: str) -> str:
    """Remove SEC SGML header — everything before the actual document content."""
    # Strategy: find where the actual 10-K content begins
    # Pattern 1: After </SEC-HEADER>
    m = re.search(r'</SEC-HEADER>', text)
    if m:
        text = text[m.end():]

    # Pattern 2: Skip the 10-K reference line
    m = re.search(r'10-K\s+\d+\s+\S+\.htm\s+10-K', text)
    if m:
        text = text[m.end():]

    # Pattern 3: Skip until after the cover page boilerplate
    # (the long block of check marks, registrant info, etc.)
    # Find "PART I" or "Item 1" as the real start
    m = re.search(r'\n\s*(PART\s+I\b[^IV])', text, re.IGNORECASE)
    if m:
        text = text[m.start():]

    return text.strip()


def extract_item1(text: str) -> str:
    """Extract Item 1 (Business) section from 10-K text."""
    # Find start of Item 1
    item1_start = None
    for pat in [
        r'(?:^|\n)\s*(?:ITEM|Item)\s*1[\.\s\-–—:]+\s*(?:BUSINESS|Business|GENERAL|General)',
        r'(?:^|\n)\s*(?:ITEM|Item)\s*1[\.\s]+[A-Z]',
        r'(?:^|\n)\s*ITEM\s+1\b(?!\s*[0-9A])',
    ]:
        m = re.search(pat, text)
        if m:
            item1_start = m.start()
            break

    if item1_start is None:
        # Fallback: use first 100K chars after header
        return text[:100000]

    # Find end of Item 1 (start of Item 1A or Item 2)
    search_from = item1_start + 100  # skip past the "Item 1" header itself
    item1_end = len(text)
    for pat in [
        r'(?:^|\n)\s*(?:ITEM|Item)\s*1A[\.\s\-–—:]',
        r'(?:^|\n)\s*(?:ITEM|Item)\s*2[\.\s\-–—:]',
        r'(?:^|\n)\s*PART\s+II\b',
    ]:
        m = re.search(pat, text[search_from:])
        if m:
            candidate = search_from + m.start()
            if candidate < item1_end:
                item1_end = candidate

    section = text[item1_start:item1_end]
    # Cap at 200K chars to be safe
    return section[:200000]


def is_boilerplate(text: str) -> bool:
    """Check if text is SEC boilerplate, not actual company content."""
    text_lower = text.lower()
    matches = sum(1 for p in SKIP_PATTERNS if re.search(p, text_lower))
    return matches >= 2


def clean_text(text: str) -> str:
    """Clean extracted text for display."""
    text = text.strip()
    # Remove leading section markers
    text = re.sub(r'^(?:ITEM\s*1[.\s]*(?:BUSINESS)?[.\s\-–—:]*)', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^(?:GENERAL|OVERVIEW|PART\s+I)[.\s\-–—:]*', '', text, flags=re.IGNORECASE).strip()
    # Remove CIK/filing references
    text = re.sub(r'\b\d{10}-\d{2}-\d{6}\b', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    # Ensure ends with period
    if text and not text.endswith('.'):
        text += '.'
    return text[:500]


def extract_sentence_around(text: str, match_start: int, match_end: int) -> str:
    """Extract a full sentence around a regex match position."""
    # Look back for sentence start
    sent_start = text.rfind('.', max(0, match_start - 200), match_start)
    sent_start = sent_start + 1 if sent_start >= 0 else max(0, match_start - 100)

    # Look forward for sentence end
    sent_end = text.find('.', match_end)
    if sent_end < 0 or sent_end - match_end > 300:
        sent_end = min(match_end + 200, len(text))
    else:
        sent_end += 1  # include the period

    return text[sent_start:sent_end].strip()


def is_hcm_context(window: str) -> bool:
    """Check if window is in an HCM/HR section (should be deprioritized)."""
    window_lower = window.lower()
    return sum(1 for m in HCM_MARKERS if m in window_lower) >= 2


def score_window(window: str, position: int = 0, total_len: int = 1) -> tuple:
    """Score a text window for mission-statement likelihood.
    Returns (score, tier, matched_keyword, evidence_text).
    position/total_len used to give a slight bonus to earlier text.
    """
    window_lower = window.lower()
    window_clean = re.sub(r'\s+', ' ', window)

    # Skip boilerplate
    if is_boilerplate(window):
        return (0, 0, "", "")

    # HCM penalty: reduce score for Tier 2/3 matches in HCM sections
    hcm_penalty = 0.35 if is_hcm_context(window) else 0

    # Position bonus: earlier text gets a small boost (max 0.05)
    pos_bonus = 0.05 * (1 - position / max(total_len, 1))

    # Tier 1: regex patterns for explicit mission (no HCM penalty for explicit)
    for pat, base_score in TIER1_PATTERNS:
        m = re.search(pat, window_lower)
        if m:
            evidence = extract_sentence_around(window_clean, m.start(), m.end())
            if not is_boilerplate(evidence):
                # Only apply HCM penalty if the mission keyword is generic
                penalty = hcm_penalty if "mission" not in pat else 0
                return (base_score + pos_bonus - penalty, 1, pat[:30], evidence)

    # Tier 2: implied purpose keywords
    for keyword, base_score in TIER2_KEYWORDS:
        idx = window_lower.find(keyword)
        if idx >= 0:
            evidence = extract_sentence_around(window_clean, idx, idx + len(keyword))
            if not is_boilerplate(evidence):
                return (base_score + pos_bonus - hcm_penalty, 2, keyword, evidence)

    # Tier 3: business description
    for keyword, base_score in TIER3_KEYWORDS:
        idx = window_lower.find(keyword)
        if idx >= 0:
            evidence = extract_sentence_around(window_clean, idx, idx + len(keyword))
            if not is_boilerplate(evidence):
                return (base_score + pos_bonus - hcm_penalty, 3, keyword, evidence)

    return (0, 0, "", "")


def extract_mission(filepath: pathlib.Path) -> dict:
    """Extract mission statement from a single 10-K file.
    Returns dict with: mission, tier, confidence, evidence, source_quote, char_offset.
    """
    raw = filepath.read_text(encoding="utf-8", errors="ignore")

    # Stage 1: strip headers + extract Item 1
    clean = strip_sec_header(raw)
    item1 = extract_item1(clean)

    if not item1 or len(item1.strip()) < 100:
        return {"mission": "", "tier": 0, "confidence": 0, "evidence": "", "source_quote": ""}

    # Normalize whitespace for searching
    text = re.sub(r'\s+', ' ', item1)

    # Stage 2: sliding window search
    WINDOW_SIZE = 600
    STEP = 250
    candidates = []

    text_len = min(len(text), 150000)
    for i in range(0, text_len, STEP):
        window = text[i:i + WINDOW_SIZE]
        score, tier, keyword, evidence = score_window(window, i, text_len)
        if score > 0:
            candidates.append({
                "score": score,
                "tier": tier,
                "keyword": keyword,
                "evidence": evidence,
                "offset": i,
            })

    if not candidates:
        return {"mission": "", "tier": 0, "confidence": 0, "evidence": "", "source_quote": ""}

    # Stage 3: select best candidate
    # Prefer: tier 1 > tier 2 > tier 3, then by score, then by position (earlier = better)
    candidates.sort(key=lambda c: (-c["tier"] * 10 + c["score"], -c["offset"]))
    # Actually: sort by tier (ascending = higher tier first), then score (descending)
    candidates.sort(key=lambda c: (c["tier"], -c["score"], c["offset"]))

    best = candidates[0]
    mission_text = clean_text(best["evidence"])

    return {
        "mission": mission_text,
        "tier": best["tier"],
        "confidence": round(best["score"], 2),
        "evidence": best["evidence"][:500],
        "source_quote": best["keyword"],
        "char_offset": best["offset"],
    }


def load_kg_missions() -> dict:
    """Load existing missions from TTL KG files."""
    kg_missions = {}
    if not TTL_DIR.exists():
        return kg_missions

    for ttl_file in TTL_DIR.glob("*.ttl"):
        if ttl_file.name == "schema.ttl":
            continue
        content = ttl_file.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r'sec:hasMission\s+"([^"]+)"', content)
        if m:
            mission = m.group(1).strip()
            if mission and len(mission) > 10:
                # Try to find CIK in the file
                cik_match = re.search(r'company_(\d+)', content)
                if cik_match:
                    cik = cik_match.group(1).zfill(10)
                    kg_missions[cik] = mission
    return kg_missions


def match_file_to_cik(filepath: pathlib.Path) -> str:
    """Extract CIK from filename like cleaned_10-K_0000002488-25-000012.txt."""
    accession = filepath.stem.replace("cleaned_10-K_", "")
    parts = accession.split("-")
    if parts:
        return parts[0]  # already zero-padded
    return ""


def run():
    t0 = time.time()
    print("=" * 70)
    print("  Mission Statement Extractor — SEC 10-K Pipeline")
    print("=" * 70)

    # Load existing registry
    if REGISTRY.exists():
        with open(REGISTRY, encoding="utf-8") as f:
            registry = json.load(f)
    else:
        print("ERROR: companies_registry.json not found!")
        return

    # Load KG missions (these are manually curated / high quality)
    kg_missions = load_kg_missions()
    print(f"\n[1] Loaded {len(kg_missions)} missions from KG (TTL files)")

    # Process each 10-K file
    if not SEC_DIR.exists():
        print(f"ERROR: {SEC_DIR} not found!")
        return

    files = sorted(SEC_DIR.glob("cleaned_10-K_*.txt"))
    print(f"[2] Found {len(files)} 10-K files to process\n")

    results = {}
    tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for filepath in files:
        cik = match_file_to_cik(filepath)
        if not cik:
            continue

        # Check if this CIK is in our registry
        if cik not in registry:
            continue

        company_name = registry[cik]["name"]

        # Priority 1: Use KG mission if available (manually curated)
        if cik in kg_missions:
            results[cik] = {
                "mission": kg_missions[cik],
                "tier": 1,
                "confidence": 1.0,
                "evidence": kg_missions[cik],
                "source": "kg",
                "source_quote": "sec:hasMission",
            }
            tier_counts[1] += 1
            print(f"  [KG ] {company_name[:35]:<35s}  T1  conf=1.00  OK")
            continue

        # Priority 2: Extract from 10-K text
        result = extract_mission(filepath)

        if result["mission"]:
            results[cik] = {
                "mission": result["mission"],
                "tier": result["tier"],
                "confidence": result["confidence"],
                "evidence": result["evidence"],
                "source": "10k",
                "source_quote": result.get("source_quote", ""),
            }
            tier_counts[result["tier"]] += 1
            tier_label = f"T{result['tier']}"
            print(f"  [10K] {company_name[:35]:<35s}  {tier_label}  conf={result['confidence']:.2f}  {result['mission'][:60]}...")
        else:
            results[cik] = {
                "mission": "",
                "tier": 0,
                "confidence": 0,
                "evidence": "",
                "source": "none",
                "source_quote": "",
            }
            tier_counts[0] += 1
            print(f"  [---] {company_name[:35]:<35s}  --  NO MISSION FOUND")

    # Stage 4: Update registry
    print(f"\n[3] Updating registry...")
    updated = 0
    for cik, result in results.items():
        if cik not in registry:
            continue

        old_mission = registry[cik]["overview"].get("mission", "")
        new_mission = result["mission"]

        # Update if: new mission is better, or old mission was garbage
        old_is_garbage = old_mission and any(
            x in old_mission.lower()
            for x in ['sgml', 'accession', 'acceptance', 'conformed',
                       'table of contents', 'securities and exchange',
                       'form 10-k', 'pursuant', 'registrant', 'check mark',
                       'commission file', 'annual report', 'washington']
        )

        if new_mission and (not old_mission or old_is_garbage):
            registry[cik]["overview"]["mission"] = new_mission
            registry[cik]["overview"]["mission_source"] = result["source"]
            registry[cik]["overview"]["mission_tier"] = result["tier"]
            registry[cik]["overview"]["mission_confidence"] = result["confidence"]
            registry[cik]["overview"]["mission_evidence"] = result["evidence"]
            updated += 1

            # Also update scores
            registry[cik]["scores"]["Mission Clarity"] = (
                3 if len(new_mission) > 20 else (1 if new_mission else 0)
            )

    # Write updated registry
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Companies processed:  {len(results)}")
    print(f"  Tier 1 (explicit):    {tier_counts[1]}")
    print(f"  Tier 2 (implied):     {tier_counts[2]}")
    print(f"  Tier 3 (description): {tier_counts[3]}")
    print(f"  No mission found:     {tier_counts[0]}")
    print(f"  Registry updated:     {updated} companies")
    print(f"  Time:                 {elapsed:.1f}s")
    print(f"  Output:               {OUTPUT_FILE}")
    print(f"{'=' * 70}")

    # Show companies that still have no mission
    no_mission = [
        (cik, registry[cik]["name"])
        for cik in registry
        if not registry[cik]["overview"].get("mission")
    ]
    if no_mission:
        print(f"\n  Companies still without mission ({len(no_mission)}):")
        for cik, name in no_mission:
            print(f"    - {name} (CIK: {cik})")


if __name__ == "__main__":
    run()
