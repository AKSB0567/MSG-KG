import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSG-KG — Mission Statement Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import os, re, json, pathlib
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
ITEM1_DIR  = BASE_DIR / "data" / "item1"
KG_DIR         = BASE_DIR / "MSGKG"
MISSION_KG_DIR = BASE_DIR / "data" / "mission_kg"

# ── Custom CSS ───────────────────────────────────────────────────────────
css_path = BASE_DIR / "assets" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── Load Registry ────────────────────────────────────────────────────────
REGISTRY_PATH = BASE_DIR / "companies_registry.json"
_REGISTRY_DATA = {}

if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        _REGISTRY_DATA = json.load(f)

COMPANIES = {
    cik: {
        "name": info["name"],
        "sector": info.get("sector", "Unknown"),
        "fiscal_year": info.get("fiscal_year", "FY2024"),
        "cik": cik,
        "filing_id": info.get("filing_id", ""),
        "filing_date": info.get("filing_date", ""),
        "sec_link": info.get("sec_link", ""),
    }
    for cik, info in _REGISTRY_DATA.items()
}


# ═══════════════════════════════════════════════════════════════════════════
# MISSION STATEMENT EVALUATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

# Buzzword dictionary — generic terms that weaken mission statements
BUZZWORDS = {
    "world-class", "leading", "innovative", "best-in-class", "synergy",
    "cutting-edge", "next-generation", "state-of-the-art", "premier",
    "excellence", "superior", "dynamic", "robust", "leverage",
    "paradigm", "holistic", "disruptive", "transformative", "empower",
    "solutions", "optimize", "scalable", "seamless", "world-leading",
    "unparalleled", "pioneering", "groundbreaking",
}

# Stakeholder keywords
STAKEHOLDER_MAP = {
    "Customers": ["customer", "client", "consumer", "user", "buyer", "patron",
                  "member", "guest", "passenger", "policyholder", "patient"],
    "Employees": ["employee", "colleague", "team member", "workforce", "staff",
                  "people", "talent", "worker", "associate"],
    "Shareholders": ["shareholder", "investor", "stockholder", "return",
                     "value creation", "dividend"],
    "Society": ["community", "communities", "society", "world", "planet",
                "environment", "sustainability", "social", "public"],
}

# Innovation / Differentiation signals
INNOVATION_KEYWORDS = [
    "innovation", "innovate", "technology", "research", "development",
    "ai", "artificial intelligence", "machine learning", "digital",
    "patent", "proprietary", "differentiat", "unique", "first-mover",
    "breakthrough", "advanced", "pioneer",
]

# Actionability keywords (concrete, measurable)
ACTION_KEYWORDS = [
    "deliver", "provide", "build", "create", "develop", "achieve",
    "reduce", "increase", "grow", "improve", "measure", "ensure",
    "produce", "manufacture", "serve", "operate", "generate",
    "transform", "enable", "accelerate", "protect", "help",
]


EVAL_DIMENSIONS = [
    "Clarity of Purpose", "Stakeholder Focus", "Value Proposition",
    "Actionability", "Innovation Signal", "Semantic Completeness",
    "Internal Consistency", "Writing Quality",
]


def evaluate_mission(mission: str, overview: dict) -> dict:
    """
    Evaluate a mission statement across multiple quality dimensions.
    Returns dict of {dimension: {score, rating, detail}}.
    Scale: 0–4.  Ratings: Outstanding (4), Strong (3), Adequate (2), Weak (1), Missing (0)
    """
    if not mission or len(mission.strip()) < 5:
        return {dim: {"score": 0, "rating": "Missing", "detail": "No mission statement found."}
                for dim in EVAL_DIMENSIONS}

    m_lower = mission.lower()
    m_words = m_lower.split()
    word_count = len(m_words)
    results = {}

    # ── 1. Clarity of Purpose ─────────────────────────────────────────
    vague_phrases = ["strive to be", "aim to", "seek to", "committed to being",
                     "dedicated to", "aspire to"]
    vague_count = sum(1 for p in vague_phrases if p in m_lower)
    has_specific_nouns = bool(re.search(
        r'\b(product|service|solution|energy|insurance|banking|real estate|'
        r'semiconductor|computing|healthcare|food|technology|transport|'
        r'aircraft|manufacturing|retail|finance)\b', m_lower))

    if has_specific_nouns and vague_count == 0 and 8 <= word_count <= 35:
        clarity_score, clarity_rating = 4, "Outstanding"
        clarity_detail = "Clear, specific, and concise purpose statement."
    elif has_specific_nouns and vague_count <= 1:
        clarity_score, clarity_rating = 3, "Strong"
        clarity_detail = "Good specificity with minor vagueness."
    elif has_specific_nouns or vague_count <= 1:
        clarity_score, clarity_rating = 2, "Adequate"
        clarity_detail = "Some specificity but could be more focused."
    elif word_count < 5:
        clarity_score, clarity_rating = 0, "Missing"
        clarity_detail = "Too short to convey meaningful purpose."
    else:
        clarity_score, clarity_rating = 1, "Weak"
        clarity_detail = f"Generic language with {vague_count} vague phrases."
    results["Clarity of Purpose"] = {
        "score": clarity_score, "rating": clarity_rating, "detail": clarity_detail}

    # ── 2. Stakeholder Focus ──────────────────────────────────────────
    stakeholders_found = []
    for group, keywords in STAKEHOLDER_MAP.items():
        if any(kw in m_lower for kw in keywords):
            stakeholders_found.append(group)

    n_stake = len(stakeholders_found)
    if n_stake >= 3:
        s_score, s_rating = 4, "Outstanding"
        s_detail = f"Addresses {n_stake} stakeholder groups: {', '.join(stakeholders_found)}."
    elif n_stake == 2:
        s_score, s_rating = 3, "Strong"
        s_detail = f"Addresses {', '.join(stakeholders_found)}."
    elif n_stake == 1:
        s_score, s_rating = 2, "Adequate"
        s_detail = f"Focuses on {stakeholders_found[0]} only."
    else:
        s_score, s_rating = 1, "Weak"
        s_detail = "No clear stakeholder identified."
    results["Stakeholder Focus"] = {
        "score": s_score, "rating": s_rating, "detail": s_detail}

    # ── 3. Value Proposition ──────────────────────────────────────────
    value_patterns = [
        r'\b(provid|deliver|offer|creat|build|enabl|generat|produc)\w*\b',
        r'\b(quality|affordable|reliable|secure|safe|clean|efficient)\b',
        r'\b(value|benefit|impact|outcome|result|experience)\b',
    ]
    value_hits = sum(1 for pat in value_patterns if re.search(pat, m_lower))

    if value_hits >= 3:
        v_score, v_rating = 4, "Outstanding"
        v_detail = "Strong value proposition with clear delivery mechanism."
    elif value_hits == 2:
        v_score, v_rating = 3, "Strong"
        v_detail = "Good value articulation."
    elif value_hits == 1:
        v_score, v_rating = 2, "Adequate"
        v_detail = "Some value signal but could be stronger."
    else:
        v_score, v_rating = 1, "Weak"
        v_detail = "No clear value proposition articulated."
    results["Value Proposition"] = {
        "score": v_score, "rating": v_rating, "detail": v_detail}

    # ── 4. Actionability ──────────────────────────────────────────────
    action_hits = sum(1 for kw in ACTION_KEYWORDS if kw in m_lower)

    if action_hits >= 3:
        a_score, a_rating = 4, "Outstanding"
        a_detail = f"Highly actionable with {action_hits} concrete verbs."
    elif action_hits == 2:
        a_score, a_rating = 3, "Strong"
        a_detail = "Good actionability."
    elif action_hits == 1:
        a_score, a_rating = 2, "Adequate"
        a_detail = "Somewhat actionable."
    else:
        a_score, a_rating = 1, "Weak"
        a_detail = "Abstract — no concrete action verbs."
    results["Actionability"] = {
        "score": a_score, "rating": a_rating, "detail": a_detail}

    # ── 5. Innovation Signal ──────────────────────────────────────────
    innov_hits = sum(1 for kw in INNOVATION_KEYWORDS if kw in m_lower)

    if innov_hits >= 3:
        i_score, i_rating = 4, "Outstanding"
        i_detail = "Strong innovation and differentiation signals."
    elif innov_hits == 2:
        i_score, i_rating = 3, "Strong"
        i_detail = "Good innovation emphasis."
    elif innov_hits == 1:
        i_score, i_rating = 2, "Adequate"
        i_detail = "Some innovation signal."
    else:
        i_score, i_rating = 1, "Weak"
        i_detail = "No innovation or differentiation language."
    results["Innovation Signal"] = {
        "score": i_score, "rating": i_rating, "detail": i_detail}

    # ── 6. Semantic Completeness ──────────────────────────────────────
    # Check for presence of key semantic components in the mission
    addressed = [g for g, kws in STAKEHOLDER_MAP.items()
                 if any(kw in m_lower for kw in kws)]
    components = {
        "Purpose (Why)": bool(re.search(r'\b(purpose|mission|goal|aim|vision)\b', m_lower)),
        "Activity (What)": bool(re.search(r'\b(provide|deliver|build|create|develop|produce|serve|help|enable)\b', m_lower)),
        "Audience (Who)": len(addressed) > 0,
        "Differentiator (How)": bool(re.search(r'\b(innovat|unique|leading|best|premier|advanced|superior|quality)\b', m_lower)),
        "Domain (Where)": bool(re.search(r'\b(global|world|nation|region|industr|market|sector)\b', m_lower)),
    }
    n_present = sum(components.values())
    missing_names = [k for k, v in components.items() if not v]

    if n_present >= 5:
        sc_score, sc_rating = 4, "Outstanding"
        sc_detail = "All semantic components present."
    elif n_present >= 4:
        sc_score, sc_rating = 3, "Strong"
        sc_detail = f"Missing: {', '.join(missing_names)}."
    elif n_present >= 3:
        sc_score, sc_rating = 2, "Adequate"
        sc_detail = f"Missing: {', '.join(missing_names)}."
    elif n_present >= 1:
        sc_score, sc_rating = 1, "Weak"
        sc_detail = f"Missing: {', '.join(missing_names)}."
    else:
        sc_score, sc_rating = 0, "Missing"
        sc_detail = "No recognizable semantic components."
    results["Semantic Completeness"] = {
        "score": sc_score, "rating": sc_rating, "detail": sc_detail}

    # ── 7. Internal Consistency ───────────────────────────────────────
    conflicting_pairs = [
        ("low cost", "premium"), ("affordable", "luxury"),
        ("global", "local"), ("focused", "diversified"),
        ("conservative", "aggressive"), ("stable", "disruptive"),
    ]
    conflicts = [(a, b) for a, b in conflicting_pairs
                 if a in m_lower and b in m_lower]

    objectives = overview.get("objectives", [])
    capabilities = overview.get("capabilities", [])

    # Also check buzzword contamination as mild inconsistency
    buzzwords_found = [w for w in BUZZWORDS if w in m_lower]

    if not conflicts and not buzzwords_found and (objectives or capabilities):
        c_score, c_rating = 4, "Outstanding"
        c_detail = "No contradictions; clean, consistent messaging."
    elif not conflicts and len(buzzwords_found) <= 1:
        c_score, c_rating = 3, "Strong"
        c_detail = "No contradictions detected."
    elif not conflicts:
        c_score, c_rating = 2, "Adequate"
        c_detail = f"Consistent but uses buzzwords: {', '.join(buzzwords_found[:3])}."
    elif len(conflicts) == 1:
        c_score, c_rating = 1, "Weak"
        c_detail = f"Potential tension: '{conflicts[0][0]}' vs '{conflicts[0][1]}'."
    else:
        c_score, c_rating = 1, "Weak"
        c_detail = f"{len(conflicts)} conflicting signals detected."
    results["Internal Consistency"] = {
        "score": c_score, "rating": c_rating, "detail": c_detail}

    # ── 8. Writing Quality ────────────────────────────────────────────
    # Assess conciseness, structure, and absence of buzzwords/filler
    n_buzz = len(buzzwords_found)
    is_concise = 8 <= word_count <= 30
    has_active_voice = bool(re.search(r'\b(we |our )\b', m_lower))
    ends_with_period = mission.strip().endswith(".")

    quality_points = 0
    if n_buzz == 0:
        quality_points += 1
    if is_concise:
        quality_points += 1
    if has_active_voice or ends_with_period:
        quality_points += 1
    if action_hits >= 1:
        quality_points += 1

    wq_map = {4: (4, "Outstanding"), 3: (3, "Strong"),
              2: (2, "Adequate"), 1: (1, "Weak"), 0: (0, "Missing")}
    w_score, w_rating = wq_map[quality_points]
    details = []
    if n_buzz > 0:
        details.append(f"{n_buzz} buzzwords")
    if not is_concise:
        details.append(f"{'too long' if word_count > 30 else 'too short'} ({word_count} words)")
    if not details:
        details.append("concise, clean prose")
    results["Writing Quality"] = {
        "score": w_score, "rating": w_rating, "detail": "; ".join(details).capitalize() + "."}

    return results


def overall_rating(eval_results: dict) -> tuple:
    """Compute overall rating from evaluation results. Returns (score, label, color)."""
    scores = [v["score"] for v in eval_results.values() if v["score"] > 0]
    if not scores:
        return 0, "Missing", "#999999"
    avg = sum(scores) / len(scores)
    if avg >= 3.5:
        return avg, "Outstanding", "#1B5E20"
    elif avg >= 2.5:
        return avg, "Strong", "#2E7D32"
    elif avg >= 1.5:
        return avg, "Adequate", "#F57F17"
    elif avg >= 0.5:
        return avg, "Weak", "#E65100"
    else:
        return avg, "Missing", "#B71C1C"


RATING_COLORS = {
    "Outstanding": "#1B5E20",
    "Strong": "#2E7D32",
    "Adequate": "#F57F17",
    "Weak": "#E65100",
    "Missing": "#B71C1C",
}


# ═══════════════════════════════════════════════════════════════════════════
# EVIDENCE EXTRACTION — find mission source in 10-K text
# ═══════════════════════════════════════════════════════════════════════════

def _find_company_10k_file(company_cik: str) -> pathlib.Path | None:
    """Find the cleaned 10-K text file for a company by matching CIK in filename."""
    sec_text_dir = KG_DIR / "data" / "data"
    if not sec_text_dir.exists():
        return None
    # Filenames: cleaned_10-K_0000002488-25-000012.txt — CIK is first part of accession
    cik_stripped = company_cik.lstrip("0")
    for txt_file in sec_text_dir.glob("cleaned_10-K_*.txt"):
        # Extract CIK from filename: cleaned_10-K_XXXXXXXXXX-YY-ZZZZZZ.txt
        accession = txt_file.stem.replace("cleaned_10-K_", "")
        file_cik = accession.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            return txt_file
    return None


def find_mission_evidence(company_cik: str, mission: str) -> list[dict]:
    """Find text passages from the company's own 10-K filing that support the mission."""
    if not mission or len(mission) < 10:
        return []

    evidence = []
    mission_lower = mission.lower()
    mission_words = re.findall(r'\b[a-z]{4,}\b', mission_lower)
    key_phrases = set(mission_words) - {
        "the", "and", "for", "that", "with", "from", "are", "our",
        "has", "have", "been", "will", "their", "this", "not", "but",
        "which", "into", "also", "more", "would", "could", "about",
    }

    txt_file = _find_company_10k_file(company_cik)
    if txt_file:
        raw = txt_file.read_text(encoding="utf-8", errors="ignore")
        # Strip SEC header
        hdr = raw.find("</SEC-HEADER>")
        if hdr > 0:
            raw = raw[hdr + len("</SEC-HEADER>"):]
        raw_norm = re.sub(r'\s+', ' ', raw)

        # Sliding window approach (600 char windows, 300 char step)
        window_size = 600
        step = 300
        # Skip the main evidence window (already shown above)
        main_mission_pos = raw_norm.lower().find(mission_lower[:40])

        for i in range(0, min(len(raw_norm), 500000), step):
            # Skip window that overlaps with the main evidence
            if main_mission_pos >= 0 and abs(i - main_mission_pos) < window_size:
                continue
            chunk = raw_norm[i:i + window_size]
            chunk_lower = chunk.lower()
            hits = sum(1 for kw in key_phrases if kw in chunk_lower)
            if hits >= 3:
                evidence.append({
                    "chunk_id": f"10K_pos_{i}",
                    "section": "10-K Filing",
                    "page": i // 3000 + 1,
                    "text": chunk.strip(),
                    "relevance": hits / max(len(key_phrases), 1),
                    "hits": hits,
                })

    # Deduplicate (skip overlapping windows) and sort by relevance
    seen_positions = []
    unique = []
    for ev in sorted(evidence, key=lambda x: x["relevance"], reverse=True):
        pos = int(ev["chunk_id"].split("_")[-1])
        if any(abs(pos - sp) < window_size for sp in seen_positions):
            continue
        seen_positions.append(pos)
        unique.append(ev)
        if len(unique) >= 5:
            break
    return unique


def highlight_mission_in_text(text: str, mission: str) -> str:
    """Highlight mission-related keywords in text using yellow background."""
    if not mission:
        return text
    keywords = set(re.findall(r'\b[a-z]{4,}\b', mission.lower()))
    keywords -= {"that", "this", "with", "from", "have", "been", "will",
                 "their", "they", "them", "than", "into", "through",
                 "about", "which", "would", "could", "also", "more"}

    result = text
    for kw in keywords:
        pattern = re.compile(r'(\b' + re.escape(kw) + r'\w*\b)', re.IGNORECASE)
        result = pattern.sub(
            r'<mark style="background-color: #FFEB3B; padding: 1px 3px; border-radius: 2px;">\1</mark>',
            result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_rag_engine():
    try:
        import rag_engine
        return rag_engine
    except Exception:
        return None

@st.cache_resource
def load_kg_graph():
    import networkx as nx
    rag = load_rag_engine()
    if rag:
        try:
            return rag.load_kg()
        except Exception:
            pass
    return nx.DiGraph()

def clean_display_text(text: str) -> str:
    t = re.sub(r'\b\d{10}-\d{2}-\d{6}\b', '', str(text))
    t = re.sub(r'(?:inst|sec|msg|msgkg|rdf|rdfs|owl|xsd):', '', t)
    t = re.sub(r'_', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t[:200] if t else text[:200]

def format_risk_name(text: str) -> str:
    t = clean_display_text(text)
    t = re.sub(r'^Risk(?:Theme)?\s*', '', t)
    return t.strip().title() if t else "Risk"

def format_capability(text: str) -> str:
    return clean_display_text(text).title()

def get_overview(cik: str) -> dict:
    if _REGISTRY_DATA and cik in _REGISTRY_DATA:
        return _REGISTRY_DATA[cik].get("overview", {})
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background: linear-gradient(135deg, #8B0000 0%, #CD1515 50%, #B22222 100%);
     color: white; padding: 1.2rem 2rem; border-radius: 10px; margin-bottom: 1rem;
     box-shadow: 0 4px 15px rgba(139,0,0,0.3);">
    <h1 style="margin:0; font-size:1.8rem; font-weight:700; color:white;">
        MSG-KG — Mission Statement Intelligence</h1>
    <p style="margin:0.3rem 0 0 0; font-size:0.95rem; opacity:0.9; color:#FFD700;">
        Evidence-based evaluation of corporate mission statements from SEC 10-K filings across 91 S&P companies</p>
</div>
""", unsafe_allow_html=True)


if not COMPANIES:
    st.warning("No companies loaded. Run `python build_registry.py` first.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR / FILTERS
# ═══════════════════════════════════════════════════════════════════════════

col_filters, col_content = st.columns([1, 4])

with col_filters:
    st.markdown("### Company")
    st.markdown("---")

    sectors = sorted(set(COMPANIES[c]["sector"] for c in COMPANIES))
    sel_sector = st.selectbox("Sector", ["All Sectors"] + sectors,
                              label_visibility="collapsed")
    if sel_sector == "All Sectors":
        available = sorted(COMPANIES.keys(), key=lambda c: COMPANIES[c]["name"])
    else:
        available = sorted(
            [c for c in COMPANIES if COMPANIES[c]["sector"] == sel_sector],
            key=lambda c: COMPANIES[c]["name"])

    company = st.selectbox("Company", available,
                           format_func=lambda c: COMPANIES[c]["name"],
                           label_visibility="collapsed")

    meta = COMPANIES[company]
    filing_date = meta.get("filing_date", "")
    st.info(f"📅 {meta['fiscal_year']}  ·  🏭 {meta['sector'][:30]}")
    if filing_date:
        st.caption(f"Filed: {filing_date}")

    st.markdown("---")
    st.caption(f"{len(COMPANIES)} companies loaded")


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

with col_content:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 Mission Statement",
        "🔍 Evidence-Based Reasoning",
        "📊 Portfolio Comparison",
        "🕸️ Knowledge Graph",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: MISSION STATEMENT
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    overview = get_overview(company)
    mission = overview.get("mission", "")
    company_name = COMPANIES[company]["name"]

    st.markdown(f"### Mission Statement — {company_name}")

    # Mission display with overall badge and source
    eval_results = evaluate_mission(mission, overview)
    avg_score, overall_label, overall_color = overall_rating(eval_results)
    mission_source = overview.get("mission_source", "kg")
    source_label = "Knowledge Graph (Ontology-extracted)" if mission_source == "kg" else "10-K Filing (Text-extracted)"

    if mission:
        st.markdown(f"""
        <div style="background: #F0F7FF; border: 1px solid #B3D4FC; border-left: 5px solid #1565C0;
             padding: 1.2rem 1.5rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong style="font-size:1.05rem; color:#333;">Corporate Mission Statement</strong>
                <span style="background:{overall_color}; color:white; padding:4px 14px;
                      border-radius:14px; font-weight:600; font-size:0.85rem;">{overall_label}</span>
            </div>
            <blockquote style="border-left:3px solid #90CAF9; padding:0.5rem 1rem; margin:0.8rem 0;
                 font-size:1.05rem; line-height:1.6; color:#1A237E; font-style:italic;">
                "{mission}"
            </blockquote>
            <div style="font-size:0.78rem; color:#888; margin-top:4px;">
                📌 Source: {source_label}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No mission statement found for this company.")

    # Quick Evaluation Summary
    st.markdown("### Quick Evaluation Summary")

    # Score cards row
    cols = st.columns(4)
    dims = list(eval_results.keys())
    for i, dim in enumerate(dims):
        ev = eval_results[dim]
        color = RATING_COLORS.get(ev["rating"], "#999")
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:white; border:1px solid #E0E0E0; border-radius:8px;
                 padding:12px; margin-bottom:10px; border-top:4px solid {color};
                 min-height:120px;">
                <div style="font-size:0.8rem; color:#666; text-transform:uppercase;
                     letter-spacing:0.5px;">{dim}</div>
                <div style="font-size:1.5rem; font-weight:700; color:{color};
                     margin:4px 0;">{ev['score']}/4</div>
                <div style="font-size:0.75rem; color:{color}; font-weight:600;">
                    {ev['rating']}</div>
                <div style="font-size:0.75rem; color:#666; margin-top:4px;">
                    {ev['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Source file and extraction metadata
    st.markdown("---")
    st.markdown("### Source Evidence from 10-K Filing")

    txt_file = _find_company_10k_file(company)
    mission_tier = overview.get("mission_tier", 0)
    mission_conf = overview.get("mission_confidence", 0)
    mission_evidence_text = overview.get("mission_evidence", "")
    tier_labels = {1: "Tier 1 — Explicit", 2: "Tier 2 — Implied", 3: "Tier 3 — Inferred", 0: "N/A"}
    tier_colors = {1: "#2E7D32", 2: "#E65100", 3: "#F57F17", 0: "#999"}

    t_label = tier_labels.get(mission_tier, "N/A")
    t_color = tier_colors.get(mission_tier, "#999")
    source_name = txt_file.name if txt_file else overview.get("source_filename", overview.get("filing_id", "Registry"))
    st.markdown(f"""
    <div style="background:#F5F5F5; border:1px solid #DDD; border-radius:8px;
         padding:10px 14px; margin-bottom:12px; display:flex; gap:20px; align-items:center;
         flex-wrap:wrap;">
        <span style="font-size:0.82rem; color:#555;">
            <strong>Source:</strong> {source_name}</span>
        <span style="background:{t_color}; color:white; padding:2px 10px;
              border-radius:10px; font-size:0.78rem; font-weight:600;">{t_label}</span>
        <span style="font-size:0.82rem; color:#555;">
            <strong>Confidence:</strong> {mission_conf:.0%}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Find mission text in 10-K regardless of source ──
    # For KG-sourced missions, we still search the 10-K for supporting evidence
    _10k_context = ""
    _10k_mission_found = False
    if txt_file and mission:
        try:
            _raw = txt_file.read_text(encoding="utf-8", errors="ignore")
            # Strip SEC SGML header
            _hdr = _raw.find("</SEC-HEADER>")
            if _hdr > 0:
                _raw = _raw[_hdr + len("</SEC-HEADER>"):]
            _raw_norm = re.sub(r'\s+', ' ', _raw)

            # Try to find mission text (or keywords from it) in the document
            _mission_words = [w for w in re.findall(r'\b[a-z]{4,}\b', mission.lower())
                              if w not in {"that", "this", "with", "from", "have", "been",
                                           "will", "their", "they", "them", "also", "more",
                                           "would", "could", "about", "which", "into", "through"}]

            # Strategy 1: Find exact mission phrase
            _clean_m = re.sub(r'\s+', ' ', mission[:60]).strip()
            _idx = _raw_norm.lower().find(_clean_m[:40].lower())

            if _idx >= 0:
                _10k_mission_found = True
                _start = max(0, _idx - 300)
                _end = min(len(_raw_norm), _idx + len(mission) + 300)
                _10k_context = _raw_norm[_start:_end]
            else:
                # Strategy 2: Find best matching window using keyword density
                _best_score = 0
                _best_pos = 0
                _step = 200
                for _i in range(0, min(len(_raw_norm), 500000), _step):
                    _window = _raw_norm[_i:_i + 600].lower()
                    _hits = sum(1 for w in _mission_words if w in _window)
                    if _hits > _best_score:
                        _best_score = _hits
                        _best_pos = _i
                if _best_score >= 2:
                    _10k_mission_found = True
                    _10k_context = _raw_norm[max(0, _best_pos - 100):_best_pos + 800]
        except Exception:
            pass

    # Show the mission evidence chunk with highlighting
    if mission and (_10k_context or mission_evidence_text):
        evidence_text = _10k_context if _10k_context else mission_evidence_text
        highlighted_ev = highlight_mission_in_text(evidence_text[:1000], mission)
        match_label = "Exact match found" if _10k_mission_found else "Best keyword match"
        st.markdown(f"""
        <div style="background:#FFFDF0; border:1px solid #E0D8A8; border-left:5px solid #F9A825;
             border-radius:0 8px 8px 0; padding:14px; margin-bottom:12px;">
            <div style="font-size:0.78rem; color:#888; margin-bottom:8px;">
                <strong>Mission Evidence from 10-K</strong>
                <span style="margin-left:10px; background:#FFF9C4; padding:2px 8px;
                      border-radius:8px; font-size:0.72rem;">{match_label}</span></div>
            <div style="font-size:0.92rem; line-height:1.7;">{highlighted_ev}</div>
        </div>
        """, unsafe_allow_html=True)
    elif mission:
        st.info("Could not locate mission evidence in the 10-K text file.")

    # Additional supporting evidence from filing (only when 10-K files available)
    if txt_file:
        st.markdown("##### Additional Supporting Passages")
        evidence = find_mission_evidence(company, mission)
        if evidence:
            for i, ev in enumerate(evidence[:5]):
                highlighted = highlight_mission_in_text(ev["text"][:600], mission)
                relevance_pct = int(ev["relevance"] * 100)
                st.markdown(f"""
                <div style="background:#FAFAFA; border:1px solid #E0E0E0; border-radius:8px;
                     padding:12px; margin-bottom:8px;">
                    <div style="font-size:0.75rem; color:#888; margin-bottom:6px;">
                        {ev['section']} | Relevance: {relevance_pct}% | {ev['chunk_id']}</div>
                    <div style="font-size:0.88rem; line-height:1.6;">{highlighted}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No additional supporting passages found in the filing.")

    # Raw source text expander
    if txt_file and mission:
        with st.expander("View Raw 10-K Source Text (around mission)"):
            if _10k_context:
                highlighted_raw = highlight_mission_in_text(_10k_context[:2000], mission)
                st.markdown(f'<div style="font-size:0.85rem; line-height:1.7; '
                            f'background:#FFFFF0; padding:12px; border-radius:6px;">'
                            f'{highlighted_raw}</div>',
                            unsafe_allow_html=True)
            else:
                st.caption("Could not locate the mission passage in the source file.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: EVIDENCE-BASED REASONING
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    overview = get_overview(company)
    mission = overview.get("mission", "")
    company_name = COMPANIES[company]["name"]

    st.markdown(f"### Evidence-Based Reasoning — {company_name}")
    st.caption("Detailed evaluation of each mission statement dimension with supporting evidence.")

    eval_results = evaluate_mission(mission, overview)

    if mission:
        st.markdown(f'> *"{mission}"*')
        st.markdown("---")

    for dim, ev in eval_results.items():
        color = RATING_COLORS.get(ev["rating"], "#999")
        st.markdown(f"""
        <div style="background:white; border:1px solid #E0E0E0; border-radius:8px;
             padding:16px; margin-bottom:12px; border-left:5px solid {color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1rem; font-weight:600; color:#333;">{dim}</span>
                </div>
                <div style="background:{color}; color:white; padding:3px 12px;
                     border-radius:12px; font-weight:600; font-size:0.85rem;">
                    {ev['rating']} ({ev['score']}/4)
                </div>
            </div>
            <div style="font-size:0.9rem; color:#555; margin-top:8px;">
                {ev['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

    # Pattern analysis
    st.markdown("---")
    st.markdown("### Pattern Analysis")

    if mission:
        m_lower = mission.lower()

        # Buzzwords found
        found_buzz = [w for w in BUZZWORDS if w in m_lower]
        if found_buzz:
            st.markdown("**Buzzwords detected:**")
            for bw in found_buzz:
                st.markdown(f"- `{bw}` — consider replacing with a more specific term")
        else:
            st.success("No generic buzzwords detected.")

        # Stakeholders addressed
        addressed = []
        for group, keywords in STAKEHOLDER_MAP.items():
            if any(kw in m_lower for kw in keywords):
                addressed.append(group)
        missing_stakeholders = [g for g in STAKEHOLDER_MAP if g not in addressed]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Stakeholders addressed:**")
            if addressed:
                for s in addressed:
                    st.markdown(f"- {s}")
            else:
                st.caption("None explicitly mentioned.")
        with col_b:
            st.markdown("**Missing stakeholder groups:**")
            if missing_stakeholders:
                for s in missing_stakeholders:
                    st.markdown(f"- {s}")
            else:
                st.caption("All major groups addressed.")

        # Semantic completeness
        st.markdown("---")
        st.markdown("### Missing Semantic Components")
        components = {
            "Purpose (Why)": bool(re.search(r'\b(purpose|mission|goal|aim|vision)\b', m_lower)),
            "Activity (What)": bool(re.search(r'\b(provide|deliver|build|create|develop|produce|serve|help|enable)\b', m_lower)),
            "Audience (Who)": len(addressed) > 0,
            "Differentiator (How)": bool(re.search(r'\b(innovat|unique|leading|best|premier|advanced|superior|quality)\b', m_lower)),
            "Domain (Where)": bool(re.search(r'\b(global|world|nation|region|industr|market|sector)\b', m_lower)),
        }
        missing = [k for k, v in components.items() if not v]
        present = [k for k, v in components.items() if v]

        col_p, col_m = st.columns(2)
        with col_p:
            for c in present:
                st.markdown(f"- :white_check_mark: {c}")
        with col_m:
            for c in missing:
                st.markdown(f"- :x: {c}")
    else:
        st.warning("No mission statement to analyze.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: PORTFOLIO COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Portfolio Mission Statement Comparison")
    st.caption("Heatmap evaluation of mission statements across all companies in the portfolio.")

    # Compute evaluations for all companies
    all_evals = {}
    for cik in COMPANIES:
        ov = get_overview(cik)
        m = ov.get("mission", "")
        ev = evaluate_mission(m, ov)
        avg, label, _ = overall_rating(ev)
        all_evals[cik] = {
            "name": COMPANIES[cik]["name"],
            "mission": m[:100] + ("..." if len(m) > 100 else ""),
            "overall": avg,
            "overall_label": label,
            **{dim: v["score"] for dim, v in ev.items()},
        }

    df = pd.DataFrame(all_evals.values())
    if df.empty:
        st.warning("No data available.")
    else:
        # Sort options
        sort_by = st.selectbox("Sort by", ["Overall Score", "Company Name",
                                           "Clarity of Purpose", "Stakeholder Focus",
                                           "Value Proposition", "Actionability"],
                               label_visibility="visible")
        sort_col = "overall" if sort_by == "Overall Score" else (
            "name" if sort_by == "Company Name" else sort_by)
        ascending = sort_by == "Company Name"
        df = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

        # Heatmap
        st.markdown("#### Heatmap: Mission Statement Quality")

        dims = [d for d in EVAL_DIMENSIONS if d in df.columns]

        try:
            import plotly.graph_objects as go

            if dims:
                heatmap_data = df[dims].values.tolist()
                company_labels = [str(n)[:30] for n in df["name"]]

                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=dims,
                    y=company_labels,
                    colorscale=[
                        [0.0, "#B71C1C"],   # Missing (red)
                        [0.2, "#E65100"],    # Weak (orange)
                        [0.4, "#F57F17"],    # Adequate (yellow)
                        [0.6, "#F57F17"],
                        [0.8, "#2E7D32"],    # Strong (green)
                        [1.0, "#1B5E20"],    # Outstanding (dark green)
                    ],
                    zmin=0, zmax=4,
                    text=heatmap_data,
                    texttemplate="%{text:.0f}",
                    textfont={"size": 10},
                    hovertemplate="<b>%{y}</b><br>%{x}: %{z}/4<extra></extra>",
                ))

                fig.update_layout(
                    height=max(400, len(df) * 28 + 100),
                    margin=dict(l=200, r=30, t=40, b=80),
                    xaxis=dict(side="top", tickangle=-35, tickfont=dict(size=11)),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
                    font=dict(family="Arial, sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dimension columns not found in evaluation data.")
        except ImportError:
            st.warning("Plotly not available.")

        # Table fallback
        display_cols = ["name", "overall", "overall_label"] + dims
        display_cols = [c for c in display_cols if c in df.columns]
        with st.expander("View as Table"):
            st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # Summary statistics
        st.markdown("#### Portfolio Distribution")
        rating_counts = df["overall_label"].value_counts()
        col1, col2, col3, col4, col5 = st.columns(5)
        for col, label in zip(
            [col1, col2, col3, col4, col5],
            ["Outstanding", "Strong", "Adequate", "Weak", "Missing"]
        ):
            cnt = rating_counts.get(label, 0)
            color = RATING_COLORS.get(label, "#999")
            with col:
                st.markdown(f"""
                <div style="text-align:center; padding:10px; background:white;
                     border-radius:8px; border-top:4px solid {color};">
                    <div style="font-size:2rem; font-weight:700; color:{color};">{cnt}</div>
                    <div style="font-size:0.8rem; color:#666;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        # Top and bottom performers
        st.markdown("---")
        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("#### Strongest Mission Statements")
            top5 = df.nlargest(5, "overall")
            for _, row in top5.iterrows():
                st.markdown(f"- **{row['name'][:35]}** — {row['overall']:.1f}/4 "
                            f"({row['overall_label']})")
        with col_bot:
            st.markdown("#### Weakest Mission Statements")
            bot5 = df.nsmallest(5, "overall")
            for _, row in bot5.iterrows():
                st.markdown(f"- **{row['name'][:35]}** — {row['overall']:.1f}/4 "
                            f"({row['overall_label']})")

        # Sector comparison
        st.markdown("---")
        st.markdown("#### Sector Comparison")

        df_with_sector = df.copy()
        df_with_sector["Sector"] = [COMPANIES[cik]["sector"].split(" - ")[-1][:25]
                                     if cik in COMPANIES else "Unknown"
                                     for cik in COMPANIES][:len(df)]

        # Rebuild with CIK mapping
        sector_data = []
        for cik in COMPANIES:
            ov = get_overview(cik)
            m = ov.get("mission", "")
            ev = evaluate_mission(m, ov)
            avg, label, _ = overall_rating(ev)
            sector_name = COMPANIES[cik]["sector"]
            if " - " in sector_name:
                sector_name = sector_name.split(" - ", 1)[1]
            sector_data.append({"sector": sector_name[:30], "overall": avg})

        df_sector = pd.DataFrame(sector_data)
        sector_avg = df_sector.groupby("sector")["overall"].agg(["mean", "count"]).reset_index()
        sector_avg.columns = ["Sector", "Avg Score", "Companies"]
        sector_avg = sector_avg[sector_avg["Companies"] >= 2].sort_values("Avg Score", ascending=False)

        if not sector_avg.empty:
            st.dataframe(sector_avg.style.format({"Avg Score": "{:.2f}"}),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("Not enough companies per sector for comparison.")

        # CSV export
        st.markdown("---")
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Evaluation Data (CSV)",
            data=csv_data,
            file_name="mission_evaluation_portfolio.csv",
            mime="text/csv",
        )

        # Head-to-head comparison
        st.markdown("---")
        st.markdown("#### Head-to-Head Comparison")
        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            comp1 = st.selectbox("Company A", available,
                                 format_func=lambda c: COMPANIES[c]["name"],
                                 key="comp1")
        with comp_col2:
            comp2 = st.selectbox("Company B", available,
                                 format_func=lambda c: COMPANIES[c]["name"],
                                 key="comp2", index=min(1, len(available)-1))

        if comp1 != comp2:
            ov1 = get_overview(comp1)
            ov2 = get_overview(comp2)
            ev1 = evaluate_mission(ov1.get("mission", ""), ov1)
            ev2 = evaluate_mission(ov2.get("mission", ""), ov2)

            comparison_rows = []
            for dim in dims:
                s1 = ev1[dim]["score"]
                s2 = ev2[dim]["score"]
                winner = "=" if s1 == s2 else (COMPANIES[comp1]["name"][:20] if s1 > s2
                                               else COMPANIES[comp2]["name"][:20])
                comparison_rows.append({
                    "Dimension": dim,
                    COMPANIES[comp1]["name"][:20]: f"{s1}/4 ({ev1[dim]['rating']})",
                    COMPANIES[comp2]["name"][:20]: f"{s2}/4 ({ev2[dim]['rating']})",
                    "Winner": winner,
                })
            st.dataframe(pd.DataFrame(comparison_rows),
                         use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4: KNOWLEDGE GRAPH (Multi-Schema Mission KG)
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    import networkx as nx
    import unicodedata as _ud

    company_name = COMPANIES[company]["name"]
    st.markdown(f"### Knowledge Graph — {company_name}")

    # KG Schema selector
    kg_mode = st.selectbox("KG Schema", [
        "Existing KG (Full)",
        "Mission KG: LLM-Driven (General Schema)",
        "Mission KG: Ontology-Driven (FIBO Schema)",
    ], key="kg_mode")

    def _slug(name):
        s = _ud.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        s = re.sub(r'[^\w\s-]', '', s).strip().lower()
        return re.sub(r'[-\s]+', '_', s)[:50]

    def _parse_mission_ttl(ttl_path):
        """Parse a mission KG TTL file into a NetworkX DiGraph."""
        G_m = nx.DiGraph()
        if not ttl_path.exists():
            return G_m
        content = ttl_path.read_text(encoding="utf-8", errors="ignore")

        # Extract instances and their types/properties
        # Parse subject blocks: inst:Name a Type ; prop value .
        current_subject = None
        current_type = None

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@prefix"):
                continue

            # New subject
            m = re.match(r'(inst:\S+)\s+a\s+(\S+)', line)
            if m:
                current_subject = m.group(1).replace("inst:", "")
                current_type = m.group(2).replace("msg:", "").replace("fibo-be:", "")
                G_m.add_node(current_subject, **{":LABEL": current_type})
                continue

            if current_subject and line.startswith("msg:"):
                # Property line
                prop_m = re.match(r'msg:(\S+)\s+(?:inst:(\S+)|"([^"]*)")', line)
                if prop_m:
                    prop = prop_m.group(1)
                    target = prop_m.group(2)
                    literal = prop_m.group(3)

                    if target:
                        # Object property — create edge
                        G_m.add_node(target, **{":LABEL": target.split("_")[-1] if "_" in target else "Node"})
                        G_m.add_edge(current_subject, target, relation=prop)
                    elif literal:
                        # Data property — add to node
                        G_m.nodes[current_subject][prop] = literal

            # rdfs:label
            label_m = re.match(r'rdfs:label\s+"([^"]*)"', line)
            if label_m and current_subject:
                G_m.nodes[current_subject]["label"] = label_m.group(1)

        return G_m

    # ── Node color maps per schema ──
    existing_colors = {
        "Company": "#e74c3c", "Mission": "#2ecc71", "Vision": "#27ae60",
        "StrategicObjective": "#27ae60", "Strategy": "#27ae60",
        "BusinessOperations": "#ff9800", "Filing": "#607d8b",
    }
    general_colors = {
        "Company": "#e74c3c", "MissionStatement": "#2ecc71",
    }
    ontology_colors = {
        "Company": "#e74c3c", "LegalEntity": "#e74c3c",
        "MissionStatement": "#2ecc71",
        "Stakeholder": "#2196F3", "CorporateObjective": "#FF9800",
        "CorporateValue": "#9C27B0", "BusinessCapability": "#009688",
        "BusinessDomain": "#795548",
    }

    if kg_mode == "Existing KG (Full)":
        st.caption("Mission-focused subgraph from the original SEC ontology KG.")
        G = load_kg_graph()
        node_colors = existing_colors

        # Find company nodes
        cik_stripped = company.lstrip("0")
        candidates = [
            f"company_{company}", f"Company_{company}", f"Company:{company}",
            company.upper(), company.lower(), company,
            f"company_{company.lstrip('0')}",
        ]
        if company in _REGISTRY_DATA:
            name = _REGISTRY_DATA[company].get("name", "")
            if name:
                candidates.extend([
                    f"Company_{name}", f"Company:{name}",
                    name.upper(), name,
                    f"Company_{name.upper()}", f"Company:{name.upper()}",
                ])

        company_nodes = [c for c in candidates if c in G]
        for node in G.nodes():
            node_str = str(node)
            if company in node_str and G.nodes[node].get(":LABEL") == "Company":
                if node_str not in company_nodes:
                    company_nodes.append(node_str)

        if not company_nodes:
            st.warning(f"No KG data found for {company_name} in existing KG.")
            subgraph = nx.DiGraph()
        else:
            mission_labels = {"Mission", "Vision", "StrategicObjective", "Strategy"}
            sub_nodes = set(company_nodes)

            for cn in company_nodes:
                for nb in G.neighbors(cn):
                    nb_label = G.nodes[nb].get(":LABEL", "")
                    relation = G[cn][nb].get("relation", "")
                    if (nb_label in mission_labels or
                        "mission" in relation.lower() or
                        "Mission" in str(nb)):
                        sub_nodes.add(nb)
                        for n2 in G.neighbors(nb):
                            if n2 not in company_nodes:
                                sub_nodes.add(n2)

            if len(sub_nodes) <= len(company_nodes):
                for cn in company_nodes:
                    for nb in G.neighbors(cn):
                        sub_nodes.add(nb)

            subgraph = G.subgraph(sub_nodes)

    elif kg_mode == "Mission KG: LLM-Driven (General Schema)":
        st.caption("Simple Company-Mission graph. The mission is selected via semantic scoring (simulated LLM).")
        ttl_path = MISSION_KG_DIR / "general" / f"{_slug(company_name)}.ttl"
        G = _parse_mission_ttl(ttl_path)
        subgraph = G
        node_colors = general_colors

    else:  # Ontology-Driven
        st.caption("FIBO-aligned decomposition: mission broken into Stakeholders, Values, Objectives, Capabilities, Domains.")
        ttl_path = MISSION_KG_DIR / "ontology" / f"{_slug(company_name)}.ttl"
        G = _parse_mission_ttl(ttl_path)
        subgraph = G
        node_colors = ontology_colors

    # ── Render graph ──
    if subgraph.number_of_nodes() == 0:
        st.warning("No graph data available for this company/schema combination.")
    else:
        try:
            from pyvis.network import Network
            import tempfile

            net = Network(height="600px", width="100%", bgcolor="#fafbfd",
                          font_color="#333333", directed=True)
            net.toggle_physics(True)
            net.set_options("""{
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 13, "face": "Segoe UI, sans-serif",
                             "strokeWidth": 3, "strokeColor": "#ffffff"},
                    "borderWidth": 2, "shadow": true
                },
                "edges": {
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.9}},
                    "font": {"size": 10, "align": "middle"},
                    "smooth": {"type": "curvedCW", "roundness": 0.15},
                    "width": 2
                },
                "physics": {
                    "barnesHut": {"gravitationalConstant": -3000,
                                  "springLength": 200, "springConstant": 0.04}
                },
                "interaction": {"hover": true, "tooltipDelay": 100,
                                "navigationButtons": true}
            }""")

            for node in subgraph.nodes():
                nd = G.nodes[node]
                label_type = nd.get(":LABEL", "Unknown")
                text = (nd.get("label", "") or nd.get("text", "") or
                        nd.get("name", "") or nd.get("hasMission", "") or
                        nd.get("missionText", "") or str(node))
                display = clean_display_text(text)
                if len(display) > 50:
                    display = display[:47] + "..."

                color = node_colors.get(label_type, "#95a5a6")
                size = 35 if label_type in ("Company", "LegalEntity") else (
                    28 if label_type == "MissionStatement" else 22)

                tooltip_text = nd.get("missionText", "") or nd.get("label", "") or text
                tooltip = f"[{label_type}]\n{clean_display_text(tooltip_text)[:300]}"
                net.add_node(node, label=display, color=color, size=size,
                             shape="dot", title=tooltip)

            for u, v, data in subgraph.edges(data=True):
                rel = data.get("relation", "")
                edge_color = "#e74c3c" if "mission" in rel.lower() else (
                    "#2196F3" if "stakeholder" in rel.lower() else (
                    "#9C27B0" if "value" in rel.lower() else (
                    "#FF9800" if "objective" in rel.lower() else "#888")))
                net.add_edge(u, v, label=rel, title=rel,
                             color={"color": edge_color})

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html",
                                             mode="w", encoding="utf-8") as f:
                net.save_graph(f.name)
                with open(f.name, "r", encoding="utf-8") as fr:
                    graph_html = fr.read()

            # Legend
            type_counts = {}
            for node in subgraph.nodes():
                lt = G.nodes[node].get(":LABEL", "Other")
                type_counts[lt] = type_counts.get(lt, 0) + 1

            legend_items = ""
            for lbl, cnt in sorted(type_counts.items()):
                clr = node_colors.get(lbl, "#95a5a6")
                legend_items += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
                    f'<span style="display:inline-block;width:12px;height:12px;'
                    f'border-radius:50%;background:{clr};"></span>'
                    f'<span style="font-size:12px;color:#333;">{lbl} ({cnt})</span></div>')

            schema_label = kg_mode.split(":")[0].strip() if ":" in kg_mode else kg_mode
            overlay = f"""
            <div style="position:absolute;top:12px;right:12px;z-index:999;
                 background:rgba(255,255,255,0.92);border:1px solid #ddd;
                 border-radius:8px;padding:10px 14px;">
                <h4 style="margin:0 0 6px 0;font-size:13px;color:#1a3a5c;">{schema_label}</h4>
                {legend_items}
                <div style="margin-top:6px;padding-top:6px;border-top:1px solid #eee;
                     font-size:11px;color:#888;">
                    Nodes: {subgraph.number_of_nodes()} | Edges: {subgraph.number_of_edges()}
                </div>
            </div>"""
            graph_html = graph_html.replace("</body>", overlay + "</body>")

            st.components.v1.html(graph_html, height=650, scrolling=False)

        except ImportError:
            st.warning("pyvis not installed. Showing text view.")
            for u, v, data in subgraph.edges(data=True):
                u_label = G.nodes[u].get(":LABEL", "")
                v_label = G.nodes[v].get(":LABEL", "")
                rel = data.get("relation", "")
                st.markdown(f"- **[{u_label}]** {clean_display_text(str(u))[:30]} "
                            f"-> `{rel}` -> **[{v_label}]** {clean_display_text(str(v))[:30]}")

    # ── Mission details below graph ──
    st.markdown("---")
    overview_kg = get_overview(company)
    mission_kg = overview_kg.get("mission", "")

    if mission_kg:
        tier_kg = overview_kg.get("mission_tier", 0)
        source_kg = overview_kg.get("mission_source", "unknown")
        st.markdown(f"""
        <div style="background:#E8F5E9; border:1px solid #C8E6C9; border-radius:8px;
             padding:14px; margin-bottom:12px;">
            <strong>Mission Statement:</strong> <em>"{mission_kg}"</em><br>
            <span style="font-size:0.8rem; color:#555;">
                Source: {source_kg.upper()} | Tier: {tier_kg} |
                Schema: {kg_mode.split("(")[-1].replace(")", "").strip() if "(" in kg_mode else "Full KG"}</span>
        </div>
        """, unsafe_allow_html=True)

    # Show decomposition details for ontology schema
    if "Ontology" in kg_mode and subgraph.number_of_nodes() > 0:
        st.markdown("#### FIBO-Aligned Decomposition")
        decomp = {"Stakeholders": [], "Values": [], "Objectives": [],
                  "Capabilities": [], "Domains": []}
        for node in subgraph.nodes():
            nd = G.nodes[node]
            lt = nd.get(":LABEL", "")
            label = nd.get("label", clean_display_text(str(node)))
            if lt == "Stakeholder":
                decomp["Stakeholders"].append(label)
            elif lt == "CorporateValue":
                decomp["Values"].append(label)
            elif lt == "CorporateObjective":
                decomp["Objectives"].append(label)
            elif lt == "BusinessCapability":
                decomp["Capabilities"].append(label)
            elif lt == "BusinessDomain":
                decomp["Domains"].append(label)

        cols_d = st.columns(5)
        for i, (cat, items) in enumerate(decomp.items()):
            with cols_d[i]:
                st.markdown(f"**{cat}** ({len(items)})")
                for item in items:
                    st.markdown(f"- {item}")
                if not items:
                    st.caption("None found")


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MSG-KG v3.0 — Mission Statement Intelligence | "
           "Data: SEC 10-K Filings | Finance Research")
