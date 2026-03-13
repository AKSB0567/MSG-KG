"""
Build pre-computed company registry JSON for instant app startup.
Run this ONCE after any TTL/data changes:  python build_registry.py
"""
import json, re, time, pathlib
t0 = time.time()

import rag_engine

BASE_DIR  = pathlib.Path(__file__).parent
ITEM1_DIR = BASE_DIR / "data" / "item1"

# ── Mission fallback: extract from Item 1 text files ────────────────────
MISSION_PATTERNS = [
    r'(?:our\s+)?mission\s+(?:is\s+)?(?:to\s+)?([^.]+\.)',
    r'(?:our\s+)?purpose\s+(?:is\s+)?(?:to\s+)?([^.]+\.)',
    r'we\s+(?:are\s+)?(?:committed|dedicated)\s+to\s+([^.]+\.)',
    r'(?:our\s+)?vision\s+(?:is\s+)?(?:to\s+)?([^.]+\.)',
    r'we\s+(?:strive|aim|seek)\s+to\s+([^.]+\.)',
    r'(?:our\s+)?(?:company|corporation)\s+(?:is\s+)?(?:a\s+)?(?:leading|global|premier|major|diversified)\s+([^.]+\.)',
]


def clean_mission_text(text: str) -> str:
    """Clean up mission text for display."""
    text = text.strip()
    # Remove common filing boilerplate prefixes
    for prefix in ["ITEM 1", "Item 1", "BUSINESS", "GENERAL",
                    "PART I", "Part I", "OVERVIEW", "Overview"]:
        if text.upper().startswith(prefix):
            text = text[len(prefix):].lstrip(" .-–—:,\n\r")
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


def extract_mission_from_text(text: str) -> str:
    """Extract a mission-like sentence from raw text using regex."""
    text_clean = re.sub(r'\s+', ' ', text)
    text_lower = text_clean.lower()

    # Try explicit mission patterns first
    for pat in MISSION_PATTERNS:
        m = re.search(pat, text_lower)
        if m:
            # Get the match position and extract from original text for proper casing
            start = m.start()
            # Find sentence boundaries in original
            sent_start = text_clean.rfind('.', 0, start)
            sent_start = sent_start + 1 if sent_start >= 0 else max(0, start - 50)
            sent_end = text_clean.find('.', m.end())
            sent_end = sent_end + 1 if sent_end >= 0 else m.end()
            sentence = text_clean[sent_start:sent_end].strip()
            if len(sentence) > 20:
                return clean_mission_text(sentence)

    # Fallback: find the first sentence that describes what the company does
    sentences = re.split(r'(?<=[.!?])\s+', text_clean)
    for sent in sentences:
        sent = sent.strip()
        # Skip very short, all-caps headers, or boilerplate
        if len(sent) < 40:
            continue
        if sent.upper() == sent and len(sent) < 100:
            continue
        # Skip filing metadata
        if any(skip in sent.lower() for skip in [
            "table of contents", "page", "filed with", "commission file",
            "incorporated by reference", "form 10-k", "annual report",
            "securities and exchange", "fiscal year ended"
        ]):
            continue
        return clean_mission_text(sent)
    return ""


def extract_mission_from_kg(G, company_nodes) -> str:
    """Try to extract mission from KG node attributes."""
    for cn in company_nodes:
        for n in G.neighbors(cn):
            node_data = G.nodes[n]
            label = node_data.get(":LABEL", "")

            # Direct Mission node
            if label == "Mission":
                text = node_data.get("hasMission", "") or node_data.get("text", "")
                if text:
                    return clean_mission_text(text)

            # Check BusinessOperations node for hasMission attribute
            if label == "BusinessOperations":
                mission = node_data.get("hasMission", "")
                if mission:
                    return clean_mission_text(mission)
    return ""


print("Loading KG...")
G = rag_engine.load_kg()
print(f"  KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

print("Discovering companies...")
discovered = rag_engine.discover_companies()
print(f"  Found {len(discovered)} companies")

# Load item1 text files for mission fallback
item1_texts = {}
if ITEM1_DIR.exists():
    for txt_file in ITEM1_DIR.glob("*.txt"):
        ticker = txt_file.stem.replace("_item1", "")
        item1_texts[ticker.upper()] = txt_file.read_text(encoding="utf-8", errors="ignore")

# Also load cleaned 10-K text files (matched by CIK)
SEC_TEXT_DIR = BASE_DIR / "MSGKG" / "data" / "data"
cik_texts = {}
if SEC_TEXT_DIR.exists():
    for txt_file in SEC_TEXT_DIR.glob("cleaned_10-K_*.txt"):
        parts = txt_file.stem.replace("cleaned_10-K_", "").split("-")
        if parts:
            cik = parts[0]
            cik_texts[cik] = txt_file.read_text(encoding="utf-8", errors="ignore")[:8000]
    print(f"  Loaded {len(cik_texts)} cleaned 10-K text files for mission fallback")

print("Building overviews...")
registry = {}
missions_filled = 0
for cik, info in discovered.items():
    overview = rag_engine.get_company_overview(cik)
    mission = overview.get("mission", "")
    mission_source = "kg"  # Track where mission came from

    # Clean KG-sourced mission
    if mission:
        mission = clean_mission_text(mission)

    # Fallback 1: extract from KG nodes directly
    if not mission:
        company_nodes = rag_engine._find_company_nodes(G, cik)
        mission = extract_mission_from_kg(G, company_nodes)
        if mission:
            mission_source = "kg"

    # Fallback 2: extract from Item 1 text file (if available)
    if not mission:
        for ticker, text in item1_texts.items():
            name_lower = info["name"].lower().replace(",", "").replace(".", "")
            if ticker.lower() in name_lower or name_lower[:10] in text.lower()[:500]:
                mission = extract_mission_from_text(text)
                if mission:
                    mission_source = "10k"
                    break

    # Fallback 3: extract from cleaned 10-K text files (matched by CIK)
    if not mission and cik in cik_texts:
        mission = extract_mission_from_text(cik_texts[cik])
        if mission:
            mission_source = "10k"

    registry[cik] = {
        "name": info["name"],
        "cik": cik,
        "sector": info.get("industry", "Unknown"),
        "fiscal_year": "FY2024",
        "filing_id": info.get("filing_id", ""),
        "filing_date": info.get("filing_date", ""),
        "sec_link": info.get("sec_link", ""),
        "overview": {
            "mission": mission,
            "mission_source": mission_source,
            "objectives": overview.get("objectives", []),
            "capabilities": overview.get("capabilities", []),
            "initiatives": overview.get("initiatives", []),
            "risks": overview.get("risks", []),
            "hcm_metrics": overview.get("hcm_metrics", []),
            "financial_metrics": overview.get("financial_metrics", []),
            "kg_stats": overview.get("kg_stats", {"nodes": 0, "edges": 0}),
        },
    }
    if mission and not overview.get("mission"):
        missions_filled += 1

    # Pre-compute alignment scores
    ov = registry[cik]["overview"]
    n_obj = len(ov.get("objectives", []))
    n_cap = len(ov.get("capabilities", []))
    n_init = len(ov.get("initiatives", []))
    n_fin = len(ov.get("financial_metrics", []))
    n_hcm = len(ov.get("hcm_metrics", []))
    n_risk = len(ov.get("risks", []))
    registry[cik]["scores"] = {
        "Mission Clarity": 3 if len(str(mission)) > 20 else (1 if mission else 0),
        "Vision → Strategy Linkage": min(3, n_obj),
        "Strategy → Operations Grounding": min(3, n_cap + n_init),
        "Operations → Financial Linkage": 3 if n_fin >= 50 else (2 if n_fin >= 10 else (1 if n_fin > 0 else 0)),
        "HCM Disclosure Depth": 3 if n_hcm >= 3 else (2 if n_hcm >= 1 else 1),
        "Risk Awareness & Mitigation": 3 if n_risk >= 10 else (2 if n_risk >= 3 else (1 if n_risk > 0 else 0)),
        "Initiative Specificity": 3 if n_init >= 5 else (2 if n_init >= 2 else (1 if n_init > 0 else 0)),
        "Capability Articulation": 3 if n_cap >= 3 else (2 if n_cap >= 1 else 1),
    }
    src_icon = "KG" if mission_source == "kg" else "10K"
    m_flag = "Y" if mission else "N"
    print(f"  [{src_icon:>3s}] {info['name'][:38]:<38s}  mission={m_flag}  hcm={n_hcm}")

out_path = "companies_registry.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(registry, f, indent=2, ensure_ascii=False)

print(f"\nDone! {len(registry)} companies -> {out_path} ({time.time()-t0:.1f}s)")
print(f"  Missions filled via fallback: {missions_filled}")
