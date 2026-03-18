import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSG-KG — Mission Statement Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import os, re, json, pathlib
import numpy as np
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
        "period_of_report": info.get("period_of_report", ""),
        "accepted_date": info.get("accepted_date", ""),
        "date_of_change": info.get("date_of_change", ""),
        "sec_link": info.get("sec_link", ""),
    }
    for cik, info in _REGISTRY_DATA.items()
}


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING & NLP UTILITIES (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_embedder():
    """Load sentence-transformers model (cached across reruns)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Computing mission embeddings...")
def compute_mission_embeddings(_companies_dict, _registry_data, is_pkg2: bool):
    """Compute embeddings for all missions. Returns (names, missions, embeddings, sectors, ciks)."""
    model = _load_embedder()
    names, missions, sectors, ciks = [], [], [], []
    for cik, info in _companies_dict.items():
        ov = _registry_data.get(cik, {}).get("overview", {})
        # Get active mission based on pipeline
        if is_pkg2:
            llm_ext = ov.get("llm_extraction", {})
            m = llm_ext.get("improved_mission", "") or llm_ext.get("mission", "")
        else:
            phase = ov.get("mission_improvement_phase", "")
            if phase == "phase3_business_purpose" and ov.get("business_purpose"):
                m = ov["business_purpose"]
            else:
                m = ov.get("mission", "")
        if m and len(m.strip()) > 10:
            names.append(info["name"])
            missions.append(m)
            sector = info["sector"]
            if " - " in sector:
                sector = sector.split(" - ", 1)[1]
            sectors.append(sector[:25])
            ciks.append(cik)
    embeddings = model.encode(missions, normalize_embeddings=True, show_progress_bar=False)
    return names, missions, embeddings, sectors, ciks


@st.cache_data(show_spinner="Computing similarity matrix...")
def compute_similarity_matrix(_embeddings):
    """Compute cosine similarity matrix from normalized embeddings."""
    return np.dot(_embeddings, _embeddings.T)


@st.cache_data
def compute_tfidf_by_sector(_missions_list, _sectors_list):
    """Compute TF-IDF and return per-sector top terms + sector similarity."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Custom stop words: common mission words that don't differentiate
    mission_stops = [
        "company", "corporation", "inc", "corp", "our", "we", "the", "and",
        "to", "of", "in", "for", "is", "are", "with", "that", "by", "from",
        "an", "as", "on", "its", "their", "this", "be", "or", "has", "have",
        "it", "at", "which", "also", "been", "through", "committed", "aim",
    ]

    vectorizer = TfidfVectorizer(
        max_features=500, stop_words="english",
        ngram_range=(1, 2), min_df=1, max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(_missions_list)
    feature_names = vectorizer.get_feature_names_out()

    # Per-sector top terms
    sector_set = sorted(set(_sectors_list))
    sector_terms = {}
    sector_vectors = {}
    for sec in sector_set:
        indices = [i for i, s in enumerate(_sectors_list) if s == sec]
        if not indices:
            continue
        sec_tfidf = tfidf_matrix[indices].mean(axis=0).A1
        top_idx = sec_tfidf.argsort()[-10:][::-1]
        terms = [(feature_names[i], round(sec_tfidf[i], 3)) for i in top_idx
                 if feature_names[i] not in mission_stops and sec_tfidf[i] > 0.01]
        sector_terms[sec] = terms[:8]
        sector_vectors[sec] = sec_tfidf

    # Cross-sector similarity
    sec_names = list(sector_vectors.keys())
    n = len(sec_names)
    cross_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            vi = sector_vectors[sec_names[i]]
            vj = sector_vectors[sec_names[j]]
            norm_i = np.linalg.norm(vi)
            norm_j = np.linalg.norm(vj)
            if norm_i > 0 and norm_j > 0:
                cross_sim[i, j] = np.dot(vi, vj) / (norm_i * norm_j)

    return sector_terms, sec_names, cross_sim, tfidf_matrix


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

def get_active_mission(cik: str, is_pkg2: bool) -> tuple:
    """Return (mission_text, label, is_improved) based on active pipeline.
    Pkg 1 (Text Matching): returns improved mission (current `mission` field) or original
    Pkg 2 (LLM Pipeline): returns llm_extraction.improved_mission or llm_extraction.mission
    """
    ov = get_overview(cik)

    if is_pkg2:
        llm_ext = ov.get("llm_extraction", {})
        improved = llm_ext.get("improved_mission", "")
        original = llm_ext.get("mission", "")
        if improved and improved != original:
            return improved, "llm_improved", True
        elif original:
            return original, "llm_extracted", False
        return "", "none", False
    else:
        # Pkg 1: use the current (improved) mission
        has_improvement = "mission_before_improvement" in ov
        phase = ov.get("mission_improvement_phase", "")
        if phase == "phase3_business_purpose" and ov.get("business_purpose"):
            return ov["business_purpose"], "business_purpose", has_improvement
        return ov.get("mission", ""), "mission", has_improvement


def get_pipeline_steps(cik: str, is_pkg2: bool) -> tuple:
    """Return (extraction_dict, validation_dict, improved_mission) for 3-step flow.
    extraction_dict: metadata about how the mission was found
    validation_dict: LLM validation results (or None)
    improved_mission: final mission after validation corrections (or original)
    """
    ov = get_overview(cik)

    if is_pkg2:
        llm_ext = ov.get("llm_extraction", {})
        extraction = {
            "mission": llm_ext.get("mission", ""),
            "source_type": llm_ext.get("source_type", "NONE"),
            "model": llm_ext.get("model", ""),
            "chunks_searched": llm_ext.get("chunks_searched", 0),
            "candidates_found": llm_ext.get("candidates_found", 0),
            "confidence": llm_ext.get("confidence", 0),
            "reasoning": llm_ext.get("reasoning", ""),
            "quality": llm_ext.get("quality", 0),
        }
        validation = llm_ext.get("validation") or None
        improved = llm_ext.get("improved_mission", llm_ext.get("mission", ""))
        return extraction, validation, improved
    else:
        # Pkg 1: text matching extraction
        extraction = {
            "mission": ov.get("mission_before_improvement", ov.get("mission", "")),
            "source_file": ov.get("source_filename", ""),
            "mission_tier": ov.get("mission_tier", 0),
            "mission_confidence": ov.get("mission_confidence", 0),
            "verification_status": ov.get("verification_status", ""),
            "verification_match": ov.get("verification_match", ""),
        }
        # Pkg 1 validation = llm_validate_missions.py results (stored flat in overview)
        validation = None
        if ov.get("llm_is_mission"):
            validation = {
                "is_mission": ov.get("llm_is_mission", ""),
                "actual_type": ov.get("llm_actual_type", ""),
                "quality": ov.get("llm_quality", 0),
                "quality_reason": ov.get("llm_quality_reason", ""),
                "corrected_mission": ov.get("llm_corrected_mission", ""),
                "confidence": ov.get("llm_confidence", 0),
                "issues": ov.get("llm_issues", []),
            }
        improved = ov.get("mission", "")
        return extraction, validation, improved


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
    all_sorted = sorted(COMPANIES.keys(), key=lambda c: COMPANIES[c]["name"])

    # ── Pipeline Toggle ──
    st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#1565C0; margin:0 0 4px 0;">Pipeline</p>', unsafe_allow_html=True)
    if "validation_approach" not in st.session_state:
        st.session_state.validation_approach = "Pkg 1: Text Matching"
    st.radio("val_approach_radio", ["Pkg 1: Text Matching", "Pkg 2: LLM Pipeline"],
             key="validation_approach", horizontal=True, label_visibility="collapsed")

    # ── Company dropdown ──
    st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#1565C0; margin:8px 0 4px 0;">Select Company</p>', unsafe_allow_html=True)
    company = st.selectbox("company_main", all_sorted,
                           format_func=lambda c: f"{all_sorted.index(c)+1} - {COMPANIES[c]['name']}",
                           label_visibility="collapsed")

    # ── Dynamic filter based on pipeline ──
    _is_text_matching = st.session_state.validation_approach == "Pkg 1: Text Matching"

    if _is_text_matching:
        # Text Matching: filter by verification status
        st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#1565C0; margin:8px 0 4px 0;">Verification Status</p>', unsafe_allow_html=True)
        _VS_OPTIONS = [
            "All",
            "VERIFIED_M1 — Exact + mission language",
            "VERIFIED_M2 — Match + commitment language",
            "TEXT_MATCH_NO_CONTEXT — Found, no keywords",
            "KEYWORD_WITH_CONTEXT — Keywords + context",
            "KEYWORD_ONLY — Keyword density only",
            "NO_MATCH — Not found in source",
            "NO_MISSION — No mission extracted",
        ]
        _vs_filter = st.selectbox("vs_filter", _VS_OPTIONS, label_visibility="collapsed")
        _vs_val = _vs_filter.split(" — ")[0] if _vs_filter != "All" else None

        if _vs_val:
            filtered = [c for c in all_sorted
                        if _REGISTRY_DATA.get(c, {}).get("overview", {}).get("verification_status") == _vs_val]
        else:
            filtered = None
    else:
        # Pkg 2: filter by llm_extraction.validation verdict and type
        st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#1565C0; margin:8px 0 4px 0;">LLM Verdict</p>', unsafe_allow_html=True)
        _LLM_V_OPTIONS = ["All", "YES — True mission", "PARTIAL — Mixed content", "NO — Not a mission"]
        _llm_v_filter = st.selectbox("llm_v_filter", _LLM_V_OPTIONS, label_visibility="collapsed")
        _llm_v_val = _llm_v_filter.split(" — ")[0] if _llm_v_filter != "All" else None

        st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#1565C0; margin:8px 0 4px 0;">Content Type</p>', unsafe_allow_html=True)
        _LLM_T_OPTIONS = ["All", "MISSION", "STRATEGY", "MIXED", "OPERATIONS", "HR", "VISION", "FINANCIAL", "REGULATORY"]
        _llm_t_filter = st.selectbox("llm_t_filter", _LLM_T_OPTIONS, label_visibility="collapsed")
        _llm_t_val = _llm_t_filter if _llm_t_filter != "All" else None

        def _get_pkg2_validation(cik):
            return _REGISTRY_DATA.get(cik, {}).get("overview", {}).get("llm_extraction", {}).get("validation", {}) or {}

        filtered = all_sorted
        if _llm_v_val:
            filtered = [c for c in filtered
                        if _get_pkg2_validation(c).get("is_mission") == _llm_v_val]
        if _llm_t_val:
            filtered = [c for c in filtered
                        if _get_pkg2_validation(c).get("actual_type") == _llm_t_val]
        if not _llm_v_val and not _llm_t_val:
            filtered = None

    # Show filtered company list
    if filtered is not None:
        st.markdown(f'<p style="font-size:0.75rem; color:#666; margin:2px 0;">{len(filtered)} companies</p>', unsafe_allow_html=True)
        if filtered:
            company = st.selectbox("filtered_company", filtered,
                                   format_func=lambda c: COMPANIES[c]["name"],
                                   label_visibility="collapsed")

    # ── Company details card ──
    meta = COMPANIES[company]
    _ov_sidebar = _REGISTRY_DATA.get(company, {}).get("overview", {})

    # Quick validation badge for sidebar
    if _is_text_matching:
        _sb_status = _ov_sidebar.get("verification_status", "N/A")
        _sb_match = _ov_sidebar.get("verification_match", "none")
        _sb_badge_colors = {"VERIFIED_M1": "#2E7D32", "VERIFIED_M2": "#E65100",
                            "TEXT_MATCH_NO_CONTEXT": "#F9A825", "KEYWORD_WITH_CONTEXT": "#FF8F00",
                            "KEYWORD_ONLY": "#F57F17", "NO_MATCH": "#D32F2F", "NO_MISSION": "#D32F2F"}
        _sb_bc = _sb_badge_colors.get(_sb_status, "#9E9E9E")
        _sb_badge = f'<span style="background:{_sb_bc}; color:white; padding:2px 6px; border-radius:8px; font-size:0.65rem;">{_sb_status}</span>'
    else:
        _sb_pkg2_val = _ov_sidebar.get("llm_extraction", {}).get("validation", {}) or {}
        _sb_llm_is = _sb_pkg2_val.get("is_mission", "N/A")
        _sb_llm_type = _sb_pkg2_val.get("actual_type", "N/A")
        _sb_is_colors = {"YES": "#2E7D32", "PARTIAL": "#E65100", "NO": "#D32F2F"}
        _sb_ic = _sb_is_colors.get(_sb_llm_is, "#9E9E9E")
        _sb_badge = (f'<span style="background:{_sb_ic}; color:white; padding:2px 6px; border-radius:8px; font-size:0.65rem;">{_sb_llm_is}</span> '
                     f'<span style="background:#455A64; color:white; padding:2px 6px; border-radius:8px; font-size:0.65rem;">{_sb_llm_type}</span>')

    # Mission version badge
    _has_imp = "mission_before_improvement" in _ov_sidebar
    _is_m0_bp = _ov_sidebar.get("mission_improvement_phase") == "phase3_business_purpose"
    _pkg2_has_improved = _ov_sidebar.get("llm_extraction", {}).get("improved_mission", "") != _ov_sidebar.get("llm_extraction", {}).get("mission", "")
    _ver_badge = ""
    if _is_text_matching:
        if _has_imp:
            _ver_badge = '<span style="background:#00897B; color:white; padding:2px 6px; border-radius:8px; font-size:0.62rem;">Improved</span>'
    else:
        if _pkg2_has_improved and _ov_sidebar.get("llm_extraction", {}).get("improved_mission"):
            _ver_badge = '<span style="background:#00897B; color:white; padding:2px 6px; border-radius:8px; font-size:0.62rem;">LLM Improved</span>'

    _info_html = (
        f'<div style="background:#F0F4FF;border:1px solid #B3D4FC;border-radius:6px;padding:8px 10px;margin:8px 0;font-size:0.75rem;">'
        f'<div style="font-weight:700;color:#1A237E;font-size:0.85rem;margin-bottom:2px;">{meta["name"]}</div>'
        f'<div style="margin-bottom:4px;">{_sb_badge} {_ver_badge}</div>'
        f'<div style="color:#555;">{meta["sector"]}</div>'
        f'<div style="color:#555;margin-top:4px;">{meta["fiscal_year"]}</div>'
        f'<div style="margin-top:6px;border-top:1px solid #D0D8E8;padding-top:4px;">'
        f'<span style="color:#888;">Filed:</span> {meta.get("filing_date","")}<br>'
        f'<span style="color:#888;">Period:</span> {meta.get("period_of_report","")}<br>'
        f'<span style="color:#888;">Accepted:</span> {meta.get("accepted_date","")}<br>'
        f'<span style="color:#888;">Changed:</span> {meta.get("date_of_change","")}'
        f'</div></div>'
    )
    st.markdown(_info_html, unsafe_allow_html=True)
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
# TAB 1: MISSION STATEMENT (3-Step Pipeline Flow)
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    overview = get_overview(company)
    _is_pkg2 = st.session_state.validation_approach != "Pkg 1: Text Matching"
    mission, _mission_label, _is_improved = get_active_mission(company, _is_pkg2)
    _extraction, _validation, _improved = get_pipeline_steps(company, _is_pkg2)
    company_name = COMPANIES[company]["name"]

    _pkg_label = "Pkg 2: LLM Pipeline" if _is_pkg2 else "Pkg 1: Text Matching"
    st.markdown(f"### Mission Statement — {company_name}")

    # Mission display with overall badge, source, and quality code
    eval_results = evaluate_mission(mission, overview)
    avg_score, overall_label, overall_color = overall_rating(eval_results)
    mission_source = overview.get("mission_source", "kg")
    source_label = "Knowledge Graph (Ontology-extracted)" if mission_source == "kg" else "10-K Filing (Text-extracted)"

    # Mission Quality Classification
    MQ_LABELS = {
        0: ("M0", "Absent from filing", "#D32F2F"),            # Red
        1: ("M1", "Directly stated in filing", "#2E7D32"),     # Green
        2: ("M2", "Stated but requires inference", "#E65100"), # Orange
        3: ("M3", "Not stated, derived from context", "#F9A825"),# Yellow
        4: ("M4", "Present but vague", "#9E9E9E"),             # Gray
    }
    mq = overview.get("mission_quality", 0)
    mq_code, mq_label, mq_color = MQ_LABELS.get(mq, ("M0", "Not Found", "#D32F2F"))

    if mission:
        _box_bg = "#F0F7FF" if not _is_pkg2 else "#E8EAF6"
        _box_border = "#B3D4FC" if not _is_pkg2 else "#9FA8DA"
        _box_accent = "#2E7D32" if not _is_pkg2 else "#1565C0"
        _box_title = "Corporate Mission Statement"
        _quote_color = "#1A237E"
        _quote_border = "#90CAF9" if not _is_pkg2 else "#7986CB"

        if _mission_label == "business_purpose":
            _box_bg = "#F3E5F5"
            _box_border = "#CE93D8"
            _box_accent = "#7B1FA2"
            _box_title = "Business Purpose (no mission statement in filing)"
            _quote_color = "#4A148C"
            _quote_border = "#CE93D8"

        # Pipeline badge
        _pkg_badge = (f'<span style="background:{"#2E7D32" if not _is_pkg2 else "#1565C0"}; color:white; '
                      f'padding:4px 10px; border-radius:14px; font-weight:600; font-size:0.72rem;">{_pkg_label}</span>')
        _imp_badge = ""
        if _is_improved:
            _imp_badge = '<span style="background:#00897B; color:white; padding:4px 10px; border-radius:14px; font-weight:600; font-size:0.72rem;">Improved</span>'

        _mission_html = (
            f'<div style="background:{_box_bg};border:1px solid {_box_border};border-left:5px solid {_box_accent};'
            f'padding:1.2rem 1.5rem;border-radius:0 8px 8px 0;margin:1rem 0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'<strong style="font-size:1.05rem;color:#333;">{_box_title}</strong>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">'
            f'{_pkg_badge}{_imp_badge}'
            f'<span style="background:{overall_color};color:white;padding:4px 14px;border-radius:14px;font-weight:600;font-size:0.85rem;">{overall_label}</span>'
            f'</div></div>'
            f'<blockquote style="border-left:3px solid {_quote_border};padding:0.5rem 1rem;margin:0.8rem 0;'
            f'font-size:1.05rem;line-height:1.6;color:{_quote_color};font-style:italic;">'
            f'"{mission}"</blockquote>'
            f'<div style="font-size:0.78rem;color:#888;margin-top:4px;">Source: {source_label}</div>'
            f'</div>'
        )
        st.markdown(_mission_html, unsafe_allow_html=True)
    else:
        st.warning("No mission statement found for this company.")

    # Quick Evaluation Summary
    st.markdown("### Quick Evaluation Summary")

    # Check for LLM evaluation data
    _t1_llm_eval = overview.get("llm_eval", {})
    _t1_llm_dims = _t1_llm_eval.get("dimensions", {})

    # Score cards row
    cols = st.columns(4)
    dims = list(eval_results.keys())
    for i, dim in enumerate(dims):
        ev = eval_results[dim]
        color = RATING_COLORS.get(ev["rating"], "#999")
        _llm_line = ""
        if _t1_llm_dims and dim in _t1_llm_dims:
            _ld = _t1_llm_dims[dim]
            if isinstance(_ld, dict) and "score" in _ld:
                _ls = _ld["score"]
                _lc = "#1565C0"
                _llm_line = (
                    f'<div style="font-size:0.7rem;color:{_lc};margin-top:4px;'
                    f'border-top:1px dashed #ccc;padding-top:3px;">'
                    f'LLM: {_ls}/4</div>'
                )
        with cols[i % 4]:
            _card_html = (
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;'
                f'padding:12px;margin-bottom:10px;border-top:4px solid {color};min-height:120px;">'
                f'<div style="font-size:0.8rem;color:#666;text-transform:uppercase;letter-spacing:0.5px;">{dim}</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{color};margin:4px 0;">{ev["score"]}/4</div>'
                f'<div style="font-size:0.75rem;color:{color};font-weight:600;">{ev["rating"]}</div>'
                f'<div style="font-size:0.75rem;color:#666;margin-top:4px;">{ev["detail"]}</div>'
                f'{_llm_line}'
                f'</div>'
            )
            st.markdown(_card_html, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # 3-STEP PIPELINE FLOW
    # ═════════════════════════════════════════════════════════════════════
    st.markdown("---")

    # ── Horizontal stepper ──
    _step1_done = bool(_extraction.get("mission") if _is_pkg2 else _extraction.get("verification_status"))
    _step2_done = _validation is not None
    _step3_done = bool(_improved)

    _s1_color = "#2E7D32" if _step1_done else "#BDBDBD"
    _s2_color = "#1565C0" if _step2_done else "#BDBDBD"
    _s3_color = "#7B1FA2" if _step3_done else "#BDBDBD"

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0;margin:16px 0 20px 0;justify-content:center;">'
        f'<div style="background:{_s1_color};color:white;padding:8px 18px;border-radius:20px;font-weight:600;font-size:0.85rem;">1 Extract</div>'
        f'<div style="width:40px;height:3px;background:{"#2E7D32" if _step2_done else "#E0E0E0"};"></div>'
        f'<div style="background:{_s2_color};color:white;padding:8px 18px;border-radius:20px;font-weight:600;font-size:0.85rem;">2 Validate</div>'
        f'<div style="width:40px;height:3px;background:{"#1565C0" if _step3_done else "#E0E0E0"};"></div>'
        f'<div style="background:{_s3_color};color:white;padding:8px 18px;border-radius:20px;font-weight:600;font-size:0.85rem;">3 Final</div>'
        f'</div>',
        unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: EXTRACTION
    # ─────────────────────────────────────────────────────────────────────
    if _is_pkg2:
        # Pkg 2: LLM extraction details
        _ext_mission = _extraction.get("mission", "")
        _ext_model = _extraction.get("model", "Qwen2.5-7B")
        _ext_src = _extraction.get("source_type", "NONE")
        _ext_chunks = _extraction.get("chunks_searched", 0)
        _ext_cands = _extraction.get("candidates_found", 0)
        _ext_conf = _extraction.get("confidence", 0)
        _ext_reasoning = _extraction.get("reasoning", "")

        _src_colors = {"EXPLICIT": "#2E7D32", "IMPLIED": "#E65100", "DERIVED": "#F57F17", "NONE": "#D32F2F"}
        _ext_sc = _src_colors.get(_ext_src, "#999")
        _ext_conf_c = "#2E7D32" if _ext_conf >= 70 else "#E65100" if _ext_conf >= 40 else "#D32F2F"

        st.markdown(
            '<div style="background:#E3F2FD;border:1px solid #90CAF9;border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
            '<strong style="color:#1565C0;">Step 1: LLM Extraction</strong>'
            f'<span style="font-size:0.78rem;color:#555;margin-left:8px;">'
            f'Qwen2.5-7B-Instruct extracts mission from raw 10-K Item 1 text chunks.</span>'
            '</div>',
            unsafe_allow_html=True)

        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {_ext_sc};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Source Type</div>'
                f'<div style="font-size:1.0rem;font-weight:700;color:{_ext_sc};margin:4px 0;">{_ext_src}</div>'
                f'<div style="font-size:0.68rem;color:#666;">how found</div></div>',
                unsafe_allow_html=True)
        with ec2:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid #1565C0;text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Chunks</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:#1565C0;margin:4px 0;">{_ext_chunks}</div>'
                f'<div style="font-size:0.68rem;color:#666;">searched</div></div>',
                unsafe_allow_html=True)
        with ec3:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid #FF8F00;text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Candidates</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:#FF8F00;margin:4px 0;">{_ext_cands}</div>'
                f'<div style="font-size:0.68rem;color:#666;">found</div></div>',
                unsafe_allow_html=True)
        with ec4:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {_ext_conf_c};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Confidence</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{_ext_conf_c};margin:4px 0;">{_ext_conf}%</div>'
                f'<div style="font-size:0.68rem;color:#666;">{_ext_model}</div></div>',
                unsafe_allow_html=True)

        if _ext_mission:
            st.markdown(
                f'<div style="background:#F5F5F5;border:1px solid #E0E0E0;border-radius:6px;padding:10px 14px;margin:8px 0;">'
                f'<span style="font-size:0.78rem;color:#888;">Extracted mission:</span>'
                f'<div style="font-size:0.92rem;line-height:1.5;color:#333;margin-top:4px;font-style:italic;">"{_ext_mission}"</div>'
                f'</div>',
                unsafe_allow_html=True)
        if _ext_reasoning:
            st.caption(f"Reasoning: {_ext_reasoning}")

    else:
        # Pkg 1: Text matching extraction
        txt_file = _find_company_10k_file(company)
        v_status = _extraction.get("verification_status", "N/A")
        v_match = _extraction.get("verification_match", "N/A")
        mission_tier = _extraction.get("mission_tier", 0)
        mission_conf = _extraction.get("mission_confidence", 0)

        st.markdown(
            '<div style="background:#E8F5E9;border:1px solid #A5D6A7;border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
            '<strong style="color:#2E7D32;">Step 1: Text Matching Extraction</strong>'
            '<span style="font-size:0.78rem;color:#555;margin-left:8px;">'
            'Regex + sliding window extraction from 10-K filing text.</span>'
            '</div>',
            unsafe_allow_html=True)

        tier_labels = {1: "Tier 1 — Explicit", 2: "Tier 2 — Implied", 3: "Tier 3 — Inferred", 0: "N/A"}
        tier_colors = {1: "#2E7D32", 2: "#E65100", 3: "#F57F17", 0: "#999"}
        t_label = tier_labels.get(mission_tier, "N/A")
        t_color = tier_colors.get(mission_tier, "#999")

        v_method_labels = {
            "VERIFIED_M1": ("Exact match + explicit mission language", "#2E7D32"),
            "VERIFIED_M2": ("Exact/partial match + commitment language", "#E65100"),
            "TEXT_MATCH_NO_CONTEXT": ("Found in text, no mission keywords nearby", "#F9A825"),
            "KEYWORD_WITH_CONTEXT": ("Keyword match + mission/commitment language", "#FF8F00"),
            "KEYWORD_ONLY": ("Keyword density match only", "#F57F17"),
            "NO_MATCH": ("Not found in source file", "#D32F2F"),
            "NO_FILE": ("Source file unavailable", "#9E9E9E"),
            "NO_MISSION": ("No mission extracted", "#D32F2F"),
            "LOW_CONFIDENCE": ("Low confidence extraction", "#9E9E9E"),
        }
        v_desc, v_color = v_method_labels.get(v_status, (v_status, "#999"))
        source_name = txt_file.name if txt_file else overview.get("source_filename", overview.get("filing_id", "Registry"))
        conf_color = "#2E7D32" if mission_conf >= 0.7 else "#E65100" if mission_conf >= 0.4 else "#D32F2F"

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {v_color};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Status</div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:{v_color};margin:4px 0;">{v_status}</div>'
                f'<div style="font-size:0.68rem;color:#666;">{v_desc}</div></div>',
                unsafe_allow_html=True)
        with vc2:
            match_colors = {"exact": "#2E7D32", "partial": "#E65100", "keyword": "#F9A825", "none": "#D32F2F"}
            mc = match_colors.get(v_match, "#999")
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {mc};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Match Type</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{mc};margin:4px 0;">{v_match.upper()}</div>'
                f'<div style="font-size:0.68rem;color:#666;">in source 10-K</div></div>',
                unsafe_allow_html=True)
        with vc3:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {t_color};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Extraction Tier</div>'
                f'<div style="font-size:0.85rem;font-weight:700;color:{t_color};margin:4px 0;">{t_label}</div>'
                f'<div style="font-size:0.68rem;color:#666;">{source_name}</div></div>',
                unsafe_allow_html=True)
        with vc4:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {conf_color};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Confidence</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{conf_color};margin:4px 0;">{mission_conf:.0%}</div>'
                f'<div style="font-size:0.68rem;color:#666;">text match</div></div>',
                unsafe_allow_html=True)

        # Show extracted (pre-improvement) mission for Pkg 1
        _pkg1_extracted = _extraction.get("mission", "")
        if _pkg1_extracted:
            st.markdown(
                f'<div style="background:#F5F5F5;border:1px solid #E0E0E0;border-radius:6px;padding:10px 14px;margin:8px 0;">'
                f'<span style="font-size:0.78rem;color:#888;">Extracted mission:</span>'
                f'<div style="font-size:0.92rem;line-height:1.5;color:#333;margin-top:4px;font-style:italic;">"{_pkg1_extracted}"</div>'
                f'</div>',
                unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: LLM VALIDATION
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")

    _val_model_label = "Qwen2.5-72B-Instruct" if not _is_pkg2 else (_validation or {}).get("model", "Qwen2.5-72B-Instruct")
    _val_header_bg = "#E3F2FD" if not _is_pkg2 else "#EDE7F6"
    _val_header_border = "#90CAF9" if not _is_pkg2 else "#B39DDB"
    _val_header_color = "#1565C0" if not _is_pkg2 else "#5E35B1"

    st.markdown(
        f'<div style="background:{_val_header_bg};border:1px solid {_val_header_border};border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
        f'<strong style="color:{_val_header_color};">Step 2: LLM Validation</strong>'
        f'<span style="font-size:0.78rem;color:#555;margin-left:8px;">'
        f'{_val_model_label} evaluates whether the extraction is truly a mission statement.</span>'
        f'</div>',
        unsafe_allow_html=True)

    if _validation:
        _val_is = _validation.get("is_mission", "")
        _val_type = _validation.get("actual_type", "")
        _val_q = _validation.get("quality", 0)
        _val_conf = _validation.get("confidence", 0)
        _val_reason = _validation.get("quality_reason", "")
        _val_corrected = _validation.get("corrected_mission", "")
        _val_issues = _validation.get("issues", [])

        is_colors = {"YES": "#2E7D32", "PARTIAL": "#E65100", "NO": "#D32F2F"}
        is_c = is_colors.get(_val_is, "#999")
        type_colors = {
            "MISSION": "#2E7D32", "VISION": "#1565C0", "STRATEGY": "#E65100",
            "OPERATIONS": "#6A1B9A", "HR": "#C62828", "FINANCIAL": "#00695C",
            "MIXED": "#F57F17", "REGULATORY": "#455A64",
        }
        tc = type_colors.get(_val_type, "#999")
        _val_mq_labels = {
            0: ("M0", "Not a mission statement"),
            1: ("M1", "Directly stated as mission/purpose"),
            2: ("M2", "Commitment/goal/belief"),
            3: ("M3", "Derived from context"),
            4: ("M4", "Vague or generic"),
        }
        _val_mq_code, _val_mq_desc = _val_mq_labels.get(_val_q, ("M?", "Unknown"))
        _val_mq_color = MQ_LABELS.get(_val_q, ("", "", "#999"))[2]
        _val_conf_c = "#2E7D32" if _val_conf >= 85 else "#E65100" if _val_conf >= 60 else "#D32F2F"

        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {is_c};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Verdict</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{is_c};margin:4px 0;">{_val_is}</div>'
                f'<div style="font-size:0.68rem;color:#666;">is mission?</div></div>',
                unsafe_allow_html=True)
        with lc2:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {tc};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Type</div>'
                f'<div style="font-size:1.0rem;font-weight:700;color:{tc};margin:4px 0;">{_val_type}</div>'
                f'<div style="font-size:0.68rem;color:#666;">classification</div></div>',
                unsafe_allow_html=True)
        with lc3:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {_val_mq_color};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Quality</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:{_val_mq_color};margin:4px 0;">{_val_mq_code}</div>'
                f'<div style="font-size:0.68rem;color:#666;">{_val_mq_desc}</div></div>',
                unsafe_allow_html=True)
        with lc4:
            st.markdown(
                f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;padding:10px;'
                f'border-top:4px solid {_val_conf_c};text-align:center;min-height:90px;">'
                f'<div style="font-size:0.72rem;color:#888;text-transform:uppercase;">Confidence</div>'
                f'<div style="font-size:1.4rem;font-weight:700;color:{_val_conf_c};margin:4px 0;">{_val_conf}%</div>'
                f'<div style="font-size:0.68rem;color:#666;">validation</div></div>',
                unsafe_allow_html=True)

        if _val_reason:
            reason_bg = "#E8F5E9" if _val_is == "YES" else "#FFF3E0" if _val_is == "PARTIAL" else "#FFEBEE"
            reason_border = "#2E7D32" if _val_is == "YES" else "#E65100" if _val_is == "PARTIAL" else "#D32F2F"
            st.markdown(
                f'<div style="background:{reason_bg};border:1px solid {reason_border}33;border-left:4px solid {reason_border};'
                f'border-radius:0 6px 6px 0;padding:10px 14px;margin:12px 0;">'
                f'<strong style="font-size:0.82rem;color:{reason_border};">Assessment:</strong>'
                f'<span style="font-size:0.85rem;color:#333;margin-left:6px;">{_val_reason}</span>'
                f'</div>',
                unsafe_allow_html=True)

        if _val_issues:
            issues_html = " ".join(
                f'<span style="background:#FFF9C4;border:1px solid #F9A825;padding:2px 8px;'
                f'border-radius:10px;font-size:0.72rem;color:#795548;margin:2px;">{iss}</span>'
                for iss in _val_issues
            )
            st.markdown(
                f'<div style="margin:8px 0;">'
                f'<span style="font-size:0.78rem;color:#888;margin-right:6px;">Issues:</span>'
                f'{issues_html}</div>',
                unsafe_allow_html=True)

        if _val_corrected and _val_corrected != "NONE" and _val_is != "YES" and len(_val_corrected) >= 10:
            st.markdown(
                f'<div style="background:#F3E5F5;border:1px solid #CE93D8;border-left:4px solid #7B1FA2;'
                f'border-radius:0 6px 6px 0;padding:12px 14px;margin:12px 0;">'
                f'<strong style="font-size:0.82rem;color:#7B1FA2;">Suggested Correction:</strong>'
                f'<blockquote style="border-left:3px solid #CE93D8;padding:0.4rem 0.8rem;margin:8px 0;'
                f'font-size:0.95rem;line-height:1.5;color:#4A148C;font-style:italic;">'
                f'"{_val_corrected}"</blockquote></div>',
                unsafe_allow_html=True)
    else:
        st.info("LLM validation has not been run for this company yet.")

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: FINAL MISSION
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="background:#F3E5F5;border:1px solid #CE93D8;border-radius:8px;padding:10px 14px;margin-bottom:10px;">'
        '<strong style="color:#7B1FA2;">Step 3: Final Mission</strong>'
        '<span style="font-size:0.78rem;color:#555;margin-left:8px;">'
        'The mission after validation corrections, or validated as correct.</span>'
        '</div>',
        unsafe_allow_html=True)

    if _improved:
        _ext_original = _extraction.get("mission", "")
        _was_changed = _improved != _ext_original and _ext_original

        if _was_changed:
            _reason = ""
            if _is_pkg2:
                _reason = overview.get("llm_extraction", {}).get("improvement_reason", "")
            else:
                _reason = "LLM-improved during 4-phase pipeline"

            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown(
                    f'<div style="background:#FFEBEE;border:1px solid #FFCDD2;border-radius:8px;padding:12px;">'
                    f'<div style="font-size:0.75rem;color:#C62828;font-weight:600;margin-bottom:6px;">BEFORE (Extraction)</div>'
                    f'<div style="font-size:0.9rem;color:#333;font-style:italic;">"{_ext_original}"</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            with fc2:
                st.markdown(
                    f'<div style="background:#E8F5E9;border:1px solid #C8E6C9;border-radius:8px;padding:12px;">'
                    f'<div style="font-size:0.75rem;color:#2E7D32;font-weight:600;margin-bottom:6px;">AFTER (Final)</div>'
                    f'<div style="font-size:0.9rem;color:#333;font-style:italic;">"{_improved}"</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            if _reason:
                st.caption(f"Reason: {_reason}")
        else:
            st.markdown(
                f'<div style="background:#E8F5E9;border:1px solid #C8E6C9;border-left:4px solid #2E7D32;'
                f'border-radius:0 8px 8px 0;padding:12px 14px;">'
                f'<div style="font-size:0.78rem;color:#2E7D32;font-weight:600;margin-bottom:6px;">Validated as correct</div>'
                f'<div style="font-size:0.95rem;color:#333;font-style:italic;">"{_improved}"</div>'
                f'</div>',
                unsafe_allow_html=True)
    else:
        st.warning("No final mission available.")

    # ── Source evidence (shared) ──
    if mission:
        txt_file = _find_company_10k_file(company)
        mission_evidence_text = overview.get("mission_evidence", "")
        _10k_context = ""
        _10k_mission_found = False

        if txt_file:
            try:
                _raw = txt_file.read_text(encoding="utf-8", errors="ignore")
                _hdr = _raw.find("</SEC-HEADER>")
                if _hdr > 0:
                    _raw = _raw[_hdr + len("</SEC-HEADER>"):]
                _raw_norm = re.sub(r'\s+', ' ', _raw)
                _mission_words = [w for w in re.findall(r'\b[a-z]{4,}\b', mission.lower())
                                  if w not in {"that", "this", "with", "from", "have", "been",
                                               "will", "their", "they", "them", "also", "more",
                                               "would", "could", "about", "which", "into", "through"}]
                _clean_m = re.sub(r'\s+', ' ', mission[:60]).strip()
                _idx = _raw_norm.lower().find(_clean_m[:40].lower())
                if _idx >= 0:
                    _10k_mission_found = True
                    _start = max(0, _idx - 300)
                    _end = min(len(_raw_norm), _idx + len(mission) + 300)
                    _10k_context = _raw_norm[_start:_end]
                else:
                    _best_score = 0
                    _best_pos = 0
                    for _i in range(0, min(len(_raw_norm), 500000), 200):
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

        with st.expander("Source Evidence from 10-K"):
            if _10k_context or mission_evidence_text:
                evidence_text = _10k_context if _10k_context else mission_evidence_text
                highlighted_ev = highlight_mission_in_text(evidence_text[:1000], mission)
                match_label = "Exact match found" if _10k_mission_found else "Best keyword match"
                st.markdown(
                    f'<div style="background:#FFFDF0;border:1px solid #E0D8A8;border-left:5px solid #F9A825;'
                    f'border-radius:0 8px 8px 0;padding:14px;margin-bottom:12px;">'
                    f'<div style="font-size:0.78rem;color:#888;margin-bottom:8px;">'
                    f'<strong>10-K Filing Context</strong>'
                    f'<span style="margin-left:10px;background:#FFF9C4;padding:2px 8px;border-radius:8px;font-size:0.72rem;">{match_label}</span></div>'
                    f'<div style="font-size:0.92rem;line-height:1.7;">{highlighted_ev}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("Could not locate mission evidence in the 10-K text file.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: EVIDENCE-BASED REASONING
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    overview = get_overview(company)
    _is_pkg2_t2 = st.session_state.validation_approach != "Pkg 1: Text Matching"
    mission, _ml_t2, _imp_t2 = get_active_mission(company, _is_pkg2_t2)
    company_name = COMPANIES[company]["name"]

    st.markdown(f"### Evidence-Based Reasoning — {company_name}")
    _approach_label_t2 = "Pkg 2: LLM Pipeline" if _is_pkg2_t2 else "Pkg 1: Text Matching"
    st.caption(f"Detailed evaluation of each mission statement dimension. Active: **{_approach_label_t2}**")

    eval_results = evaluate_mission(mission, overview)

    if mission:
        if _ml_t2 == "business_purpose":
            st.markdown(f"**Business Purpose** (no mission in filing):")
        st.markdown(f'> *"{mission}"*')
        st.markdown("---")

    for dim, ev in eval_results.items():
        color = RATING_COLORS.get(ev["rating"], "#999")
        st.markdown(
            f'<div style="background:white;border:1px solid #E0E0E0;border-radius:8px;'
            f'padding:16px;margin-bottom:12px;border-left:5px solid {color};">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div><span style="font-size:1rem;font-weight:600;color:#333;">{dim}</span></div>'
            f'<div style="background:{color};color:white;padding:3px 12px;border-radius:12px;font-weight:600;font-size:0.85rem;">'
            f'{ev["rating"]} ({ev["score"]}/4)</div></div>'
            f'<div style="font-size:0.9rem;color:#555;margin-top:8px;">{ev["detail"]}</div>'
            f'</div>',
            unsafe_allow_html=True)

    # ── LLM Evaluation Comparison (if available) ──
    _llm_eval = overview.get("llm_eval", {})
    _llm_dims = _llm_eval.get("dimensions", {})
    if _llm_dims:
        st.markdown("---")
        st.markdown("### Rule-Based vs LLM Evaluation Comparison")
        st.caption("Side-by-side comparison of rule-based keyword scoring vs Qwen2.5-72B semantic evaluation.")

        _cmp_rows = []
        for dim in EVAL_DIMENSIONS:
            rb = eval_results.get(dim, {})
            lb = _llm_dims.get(dim, {})
            rb_score = rb.get("score", 0)
            lb_score = lb.get("score", 0) if isinstance(lb, dict) else 0
            diff = lb_score - rb_score
            diff_str = f"+{diff}" if diff > 0 else str(diff) if diff != 0 else "="
            _cmp_rows.append({
                "Dimension": dim,
                "Rule-Based": f"{rb_score}/4 ({rb.get('rating', '?')})",
                "LLM (Qwen2.5)": f"{lb_score}/4",
                "Delta": diff_str,
                "LLM Reasoning": lb.get("reason", "") if isinstance(lb, dict) else "",
            })
        st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

        # Overall comparison
        rb_avg = sum(v["score"] for v in eval_results.values()) / max(len(eval_results), 1)
        llm_avg = _llm_eval.get("overall_score", 0)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rule-Based Overall", f"{rb_avg:.1f}/4")
        with c2:
            st.metric("LLM Overall", f"{llm_avg:.1f}/4", delta=f"{llm_avg - rb_avg:+.1f}")

        # LLM assessment and benchmark
        _assess = _llm_eval.get("overall_assessment", "")
        _bench = _llm_eval.get("benchmark_comparison", "")
        if _assess:
            st.markdown(f"**LLM Assessment:** {_assess}")
        if _bench:
            st.markdown(f"**Sector Benchmark:** {_bench}")

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
    _is_llm_mode_t3 = st.session_state.validation_approach != "Pkg 1: Text Matching"
    _approach_label_t3 = "Pkg 2: LLM Pipeline" if _is_llm_mode_t3 else "Pkg 1: Text Matching"
    st.caption(f"Heatmap evaluation across all companies. Active: **{_approach_label_t3}**")

    # Compute evaluations for all companies using active pipeline's missions
    all_evals = {}
    for cik in COMPANIES:
        ov = get_overview(cik)
        m, _ml_t3, _ = get_active_mission(cik, _is_llm_mode_t3)
        ev = evaluate_mission(m, ov)
        avg, label, _ = overall_rating(ev)
        all_evals[cik] = {
            "name": COMPANIES[cik]["name"],
            "mission": m[:100] + ("..." if len(m) > 100 else ""),
            "overall": avg,
            "overall_label": label,
            "source_type": _ml_t3,
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
                st.markdown(
                    f'<div style="text-align:center;padding:10px;background:white;'
                    f'border-radius:8px;border-top:4px solid {color};">'
                    f'<div style="font-size:2rem;font-weight:700;color:{color};">{cnt}</div>'
                    f'<div style="font-size:0.8rem;color:#666;">{label}</div></div>',
                    unsafe_allow_html=True)

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
            m, _, _ = get_active_mission(cik, _is_llm_mode_t3)
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
            all_ciks = sorted(COMPANIES.keys(), key=lambda c: COMPANIES[c]["name"])
            comp1 = st.selectbox("Company A", all_ciks,
                                 format_func=lambda c: COMPANIES[c]["name"],
                                 key="comp1")
        with comp_col2:
            all_ciks = sorted(COMPANIES.keys(), key=lambda c: COMPANIES[c]["name"])
            comp2 = st.selectbox("Company B", all_ciks,
                                 format_func=lambda c: COMPANIES[c]["name"],
                                 key="comp2", index=min(1, len(all_ciks)-1))

        if comp1 != comp2:
            ov1 = get_overview(comp1)
            ov2 = get_overview(comp2)
            _m1_h2h, _, _ = get_active_mission(comp1, _is_llm_mode_t3)
            _m2_h2h, _, _ = get_active_mission(comp2, _is_llm_mode_t3)
            ev1 = evaluate_mission(_m1_h2h, ov1)
            ev2 = evaluate_mission(_m2_h2h, ov2)

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

        # ═════════════════════════════════════════════════════════════════
        # MISSION SIMILARITY ANALYSIS (Embedding-based)
        # ═════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### Mission Similarity Analysis")
        st.caption("Semantic similarity between mission statements using MiniLM-L6-v2 embeddings + cosine similarity.")

        try:
            _emb_names, _emb_missions, _emb_vecs, _emb_sectors, _emb_ciks = (
                compute_mission_embeddings(
                    COMPANIES, _REGISTRY_DATA, _is_llm_mode_t3
                )
            )
            _sim_matrix = compute_similarity_matrix(_emb_vecs)
            n_emb = len(_emb_names)

            if n_emb >= 5:
                # ── Top similar pairs ──
                sim_pairs = []
                for i in range(n_emb):
                    for j in range(i + 1, n_emb):
                        sim_pairs.append((i, j, _sim_matrix[i, j]))
                sim_pairs.sort(key=lambda x: -x[2])

                st.markdown("##### Most Similar Mission Pairs")
                _sim_rows = []
                for i, j, s in sim_pairs[:10]:
                    _sim_rows.append({
                        "Company A": _emb_names[i][:30],
                        "Company B": _emb_names[j][:30],
                        "Similarity": f"{s:.3f}",
                        "Sector A": _emb_sectors[i],
                        "Sector B": _emb_sectors[j],
                        "Same Sector": "Yes" if _emb_sectors[i] == _emb_sectors[j] else "No",
                    })
                st.dataframe(pd.DataFrame(_sim_rows), use_container_width=True, hide_index=True)

                # ── Most unique missions ──
                st.markdown("##### Most Unique Missions (lowest avg similarity)")
                avg_sim = []
                for i in range(n_emb):
                    others = [_sim_matrix[i, j] for j in range(n_emb) if j != i]
                    avg_sim.append(np.mean(others))
                unique_idx = np.argsort(avg_sim)[:8]
                _uniq_rows = []
                for idx in unique_idx:
                    _uniq_rows.append({
                        "Company": _emb_names[idx][:35],
                        "Sector": _emb_sectors[idx],
                        "Avg Similarity": f"{avg_sim[idx]:.3f}",
                        "Mission": _emb_missions[idx][:100] + "...",
                    })
                st.dataframe(pd.DataFrame(_uniq_rows), use_container_width=True, hide_index=True)

                # ── Similarity Heatmap ──
                with st.expander("View Full Similarity Heatmap"):
                    import plotly.graph_objects as go
                    short_names = [n[:20] for n in _emb_names]
                    fig_sim = go.Figure(data=go.Heatmap(
                        z=_sim_matrix.tolist(),
                        x=short_names, y=short_names,
                        colorscale="RdYlGn", zmin=0, zmax=1,
                        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Similarity: %{z:.3f}<extra></extra>",
                    ))
                    fig_sim.update_layout(
                        height=max(500, n_emb * 12 + 100),
                        margin=dict(l=150, r=30, t=40, b=150),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
                    )
                    st.plotly_chart(fig_sim, use_container_width=True)

                # ── t-SNE Scatter Plot (colored by sector) ──
                st.markdown("##### Mission Landscape (t-SNE)")
                st.caption("2D projection of mission embeddings. Companies close together have semantically similar missions.")
                try:
                    from sklearn.manifold import TSNE
                    perp = min(30, max(5, n_emb // 3))
                    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                                learning_rate="auto", init="pca")
                    coords = tsne.fit_transform(_emb_vecs)

                    import plotly.express as px
                    _tsne_df = pd.DataFrame({
                        "x": coords[:, 0], "y": coords[:, 1],
                        "Company": _emb_names,
                        "Sector": _emb_sectors,
                        "Mission": [m[:80] + "..." for m in _emb_missions],
                    })
                    fig_tsne = px.scatter(
                        _tsne_df, x="x", y="y", color="Sector",
                        hover_name="Company", hover_data=["Mission"],
                        title="Mission Statement Semantic Space",
                    )
                    fig_tsne.update_layout(
                        height=550, showlegend=True,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                        legend=dict(font=dict(size=9), itemsizing="constant"),
                    )
                    fig_tsne.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
                    st.plotly_chart(fig_tsne, use_container_width=True)
                except Exception as e_tsne:
                    st.caption(f"t-SNE not available: {e_tsne}")

                # ── Cluster Analysis ──
                st.markdown("##### Mission Clusters (Hierarchical)")
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    n_clusters = min(8, max(3, n_emb // 10))
                    clust = AgglomerativeClustering(
                        n_clusters=n_clusters, metric="cosine", linkage="average"
                    )
                    labels = clust.fit_predict(_emb_vecs)

                    cluster_data = []
                    for cl in range(n_clusters):
                        members = [_emb_names[i][:25] for i in range(n_emb) if labels[i] == cl]
                        sectors_in = list(set(_emb_sectors[i] for i in range(n_emb) if labels[i] == cl))
                        # Get representative mission (closest to cluster centroid)
                        cl_indices = [i for i in range(n_emb) if labels[i] == cl]
                        cl_vecs = _emb_vecs[cl_indices]
                        centroid = cl_vecs.mean(axis=0)
                        centroid /= np.linalg.norm(centroid)
                        dists = np.dot(cl_vecs, centroid)
                        rep_idx = cl_indices[np.argmax(dists)]
                        cluster_data.append({
                            "Cluster": cl + 1,
                            "Size": len(members),
                            "Companies": ", ".join(members[:5]) + ("..." if len(members) > 5 else ""),
                            "Sectors": ", ".join(sectors_in[:3]) + ("..." if len(sectors_in) > 3 else ""),
                            "Representative": _emb_missions[rep_idx][:80] + "...",
                        })
                    st.dataframe(pd.DataFrame(cluster_data), use_container_width=True, hide_index=True)
                except Exception as e_cl:
                    st.caption(f"Clustering not available: {e_cl}")

            else:
                st.info("Need at least 5 companies with missions for similarity analysis.")

        except Exception as e_emb:
            st.warning(f"Embedding analysis unavailable: {e_emb}")

        # ═════════════════════════════════════════════════════════════════
        # SECTOR THEME ANALYSIS (TF-IDF)
        # ═════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### Sector Mission Theme Analysis")
        st.caption("TF-IDF analysis of mission keywords by sector. Shows what makes each sector's missions distinctive.")

        try:
            _emb_names2, _emb_missions2, _emb_vecs2, _emb_sectors2, _emb_ciks2 = (
                compute_mission_embeddings(
                    COMPANIES, _REGISTRY_DATA, _is_llm_mode_t3
                )
            )

            sector_terms, sec_names, cross_sim, tfidf_mx = compute_tfidf_by_sector(
                _emb_missions2, _emb_sectors2
            )

            if sector_terms:
                # ── Top terms per sector ──
                st.markdown("##### Distinctive Terms per Sector")
                _term_rows = []
                for sec in sorted(sector_terms.keys()):
                    terms = sector_terms[sec]
                    if terms:
                        kws = ", ".join(f"**{t}** ({s:.2f})" for t, s in terms[:5])
                        _term_rows.append({"Sector": sec, "Top Keywords (TF-IDF)": kws})
                if _term_rows:
                    for row in _term_rows:
                        st.markdown(
                            f'<div style="background:white;border:1px solid #E0E0E0;border-radius:6px;'
                            f'padding:8px 12px;margin-bottom:6px;">'
                            f'<strong style="color:#1565C0;">{row["Sector"]}</strong><br>'
                            f'<span style="font-size:0.85rem;color:#555;">{row["Top Keywords (TF-IDF)"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True)

                # ── Cross-sector similarity ──
                if len(sec_names) >= 3:
                    st.markdown("##### Cross-Sector Mission Similarity")
                    st.caption("Sectors using similar language in their missions. High similarity = cookie-cutter missions.")
                    import plotly.graph_objects as go
                    fig_xsec = go.Figure(data=go.Heatmap(
                        z=cross_sim.tolist(),
                        x=sec_names, y=sec_names,
                        colorscale="Blues", zmin=0, zmax=1,
                        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Theme Similarity: %{z:.3f}<extra></extra>",
                    ))
                    fig_xsec.update_layout(
                        height=max(350, len(sec_names) * 30 + 100),
                        margin=dict(l=180, r=30, t=40, b=150),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                    )
                    st.plotly_chart(fig_xsec, use_container_width=True)

                # ── Within-sector differentiation ──
                st.markdown("##### Mission Differentiation Score")
                st.caption("How unique is each company's mission WITHIN its sector? Lower = more cookie-cutter.")
                _diff_rows = []
                for i, cik in enumerate(_emb_ciks2):
                    sec = _emb_sectors2[i]
                    same_sec = [j for j in range(len(_emb_ciks2))
                                if _emb_sectors2[j] == sec and j != i]
                    if same_sec:
                        sim_matrix2 = compute_similarity_matrix(_emb_vecs2)
                        avg_intra = np.mean([sim_matrix2[i, j] for j in same_sec])
                        differentiation = round(1.0 - avg_intra, 3)
                    else:
                        differentiation = 1.0  # Only company in sector = fully unique
                    _diff_rows.append({
                        "Company": _emb_names2[i][:30],
                        "Sector": sec,
                        "Differentiation": differentiation,
                        "Same-Sector Peers": len(same_sec),
                    })
                _diff_df = pd.DataFrame(_diff_rows).sort_values("Differentiation", ascending=True)

                # Show most and least differentiated
                col_least, col_most = st.columns(2)
                with col_least:
                    st.markdown("**Most Similar to Peers** (cookie-cutter)")
                    _bottom = _diff_df[_diff_df["Same-Sector Peers"] > 0].head(8)
                    st.dataframe(_bottom[["Company", "Sector", "Differentiation"]],
                                 use_container_width=True, hide_index=True)
                with col_most:
                    st.markdown("**Most Unique in Sector**")
                    _top = _diff_df[_diff_df["Same-Sector Peers"] > 0].tail(8).iloc[::-1]
                    st.dataframe(_top[["Company", "Sector", "Differentiation"]],
                                 use_container_width=True, hide_index=True)

        except Exception as e_tfidf:
            st.warning(f"Sector theme analysis unavailable: {e_tfidf}")

        # ═════════════════════════════════════════════════════════════════
        # PACKAGE 1 vs PACKAGE 2 COMPARISON ANALYSIS
        # ═════════════════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("#### Package 1 vs Package 2 Comparison")
        st.caption("How do the two extraction pipelines agree across all 91 companies?")

        try:
            import plotly.graph_objects as go
            import plotly.express as px

            # ── Gather comparison data ──
            _cmp_rows = []
            for _cik in COMPANIES:
                _ov = get_overview(_cik)
                _tm_mission = _ov.get("mission_before_improvement", _ov.get("mission", ""))
                _llm_ext = _ov.get("llm_extraction", {})
                _llm_ext_mission = _llm_ext.get("mission", "")

                # Text matching verification
                _vs = _ov.get("verification_status", "NO_DATA")
                _vm = _ov.get("verification_match", "none")
                _tm_quality = _ov.get("mission_quality_prev", _ov.get("mission_quality", 0))
                _tm_tier = _ov.get("mission_tier", 0)

                # LLM validation verdict (from llm_validate_missions.py)
                _llm_verdict = _ov.get("llm_is_mission", "")
                _llm_type = _ov.get("llm_actual_type", "")
                _llm_quality = _ov.get("llm_quality", 0)
                _llm_conf = _ov.get("llm_confidence", 0)

                # LLM extraction (from llm_extract_missions.py)
                _llm_ext_verdict = _llm_ext.get("is_mission", "")
                _llm_ext_type = _llm_ext.get("actual_type", "")
                _llm_ext_quality = _llm_ext.get("quality", 0)
                _llm_ext_conf = _llm_ext.get("confidence", 0)
                _llm_ext_src = _llm_ext.get("source_type", "")

                # Word overlap between text-matching and LLM extraction
                _overlap = 0.0
                if _tm_mission and _llm_ext_mission:
                    _tw = set(re.findall(r'\b\w{4,}\b', _tm_mission.lower()))
                    _lw = set(re.findall(r'\b\w{4,}\b', _llm_ext_mission.lower()))
                    if _tw and _lw:
                        _overlap = len(_tw & _lw) / max(len(_tw | _lw), 1)

                _cmp_rows.append({
                    "name": COMPANIES[_cik]["name"],
                    "sector": COMPANIES[_cik]["sector"].split(" - ")[-1][:25] if " - " in COMPANIES[_cik]["sector"] else COMPANIES[_cik]["sector"][:25],
                    "tm_status": _vs,
                    "tm_match": _vm,
                    "tm_quality": _tm_quality,
                    "tm_tier": _tm_tier,
                    "tm_mission": _tm_mission[:80] if _tm_mission else "",
                    "llm_verdict": _llm_verdict if _llm_verdict else "N/A",
                    "llm_type": _llm_type if _llm_type else "N/A",
                    "llm_quality": _llm_quality,
                    "llm_conf": _llm_conf,
                    "llm_ext_verdict": _llm_ext_verdict if _llm_ext_verdict else "N/A",
                    "llm_ext_type": _llm_ext_type if _llm_ext_type else "N/A",
                    "llm_ext_quality": _llm_ext_quality,
                    "llm_ext_conf": _llm_ext_conf,
                    "llm_ext_src": _llm_ext_src if _llm_ext_src else "N/A",
                    "word_overlap": round(_overlap, 3),
                })

            _cmp_df = pd.DataFrame(_cmp_rows)

            if not _cmp_df.empty:
                # ── 1. LLM Verdict Distribution (Pie + Bar) ──
                st.markdown("##### Pkg 1 LLM Validation Verdict Distribution")
                _vc1, _vc2 = st.columns(2)

                with _vc1:
                    _verd_counts = _cmp_df["llm_verdict"].value_counts()
                    _verd_colors = {"YES": "#2E7D32", "PARTIAL": "#F57F17", "NO": "#B71C1C", "N/A": "#9E9E9E"}
                    if len(_verd_counts) > 0:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=_verd_counts.index.tolist(),
                            values=_verd_counts.values.tolist(),
                            marker=dict(colors=[_verd_colors.get(v, "#999") for v in _verd_counts.index]),
                            hole=0.4,
                            textinfo="label+value+percent",
                            textfont=dict(size=12),
                        )])
                        fig_pie.update_layout(
                            title=dict(text="LLM Verdict", font=dict(size=13)),
                            height=320, margin=dict(l=20, r=20, t=40, b=20),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                with _vc2:
                    _type_counts = _cmp_df[_cmp_df["llm_type"] != "N/A"]["llm_type"].value_counts()
                    if len(_type_counts) > 0:
                        _type_colors = {
                            "MISSION": "#2E7D32", "VISION": "#1565C0", "STRATEGY": "#F57F17",
                            "OPERATIONS": "#E65100", "HR": "#7B1FA2", "FINANCIAL": "#B71C1C",
                            "MIXED": "#FF8F00", "REGULATORY": "#455A64",
                        }
                        fig_bar_type = go.Figure(data=[go.Bar(
                            x=_type_counts.index.tolist(),
                            y=_type_counts.values.tolist(),
                            marker_color=[_type_colors.get(t, "#999") for t in _type_counts.index],
                            text=_type_counts.values.tolist(),
                            textposition="auto",
                        )])
                        fig_bar_type.update_layout(
                            title=dict(text="LLM Content Type Classification", font=dict(size=13)),
                            height=320, margin=dict(l=40, r=20, t=40, b=40),
                            xaxis=dict(tickangle=-25, tickfont=dict(size=10)),
                            yaxis=dict(title="Companies"),
                        )
                        st.plotly_chart(fig_bar_type, use_container_width=True)

                # ── 2. Text Matching vs LLM Quality Agreement ──
                st.markdown("##### Quality Rating: Pkg 1 vs Pkg 1 LLM Validation")

                _qa1, _qa2 = st.columns(2)

                with _qa1:
                    # Grouped bar chart: M0-M4 counts for each approach
                    _tm_q_counts = _cmp_df["tm_quality"].value_counts().sort_index()
                    _llm_q_counts = _cmp_df["llm_quality"].value_counts().sort_index()
                    _q_labels = ["M0", "M1", "M2", "M3", "M4"]
                    _tm_vals = [int(_tm_q_counts.get(i, 0)) for i in range(5)]
                    _llm_vals = [int(_llm_q_counts.get(i, 0)) for i in range(5)]

                    fig_q = go.Figure()
                    fig_q.add_trace(go.Bar(
                        name="Pkg 1: Text Matching", x=_q_labels, y=_tm_vals,
                        marker_color="#1565C0", text=_tm_vals, textposition="auto",
                    ))
                    fig_q.add_trace(go.Bar(
                        name="Pkg 1: LLM Validation", x=_q_labels, y=_llm_vals,
                        marker_color="#2E7D32", text=_llm_vals, textposition="auto",
                    ))
                    fig_q.update_layout(
                        barmode="group", height=320,
                        title=dict(text="Mission Quality (M0-M4)", font=dict(size=13)),
                        margin=dict(l=40, r=20, t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_q, use_container_width=True)

                with _qa2:
                    # Agreement scatter / confusion matrix
                    _agree_data = _cmp_df[(_cmp_df["llm_verdict"] != "N/A")].copy()
                    if not _agree_data.empty:
                        # Map TM status to simple categories
                        def _tm_to_verdict(status):
                            if status in ("VERIFIED_M1", "VERIFIED_M2"):
                                return "YES"
                            elif status in ("TEXT_MATCH_NO_CONTEXT", "KEYWORD_WITH_CONTEXT", "KEYWORD_ONLY"):
                                return "PARTIAL"
                            else:
                                return "NO"

                        _agree_data["tm_verdict"] = _agree_data["tm_status"].apply(_tm_to_verdict)

                        # Build confusion matrix
                        _verdict_labels = ["YES", "PARTIAL", "NO"]
                        _conf_mx = []
                        for tm_v in _verdict_labels:
                            row = []
                            for llm_v in _verdict_labels:
                                cnt = len(_agree_data[(_agree_data["tm_verdict"] == tm_v) & (_agree_data["llm_verdict"] == llm_v)])
                                row.append(cnt)
                            _conf_mx.append(row)

                        fig_conf = go.Figure(data=go.Heatmap(
                            z=_conf_mx,
                            x=[f"LLM: {v}" for v in _verdict_labels],
                            y=[f"TM: {v}" for v in _verdict_labels],
                            colorscale=[[0, "#FFFFFF"], [0.5, "#90CAF9"], [1.0, "#1565C0"]],
                            text=_conf_mx, texttemplate="%{text}",
                            textfont=dict(size=16, color="black"),
                            hovertemplate="Text Matching: %{y}<br>LLM: %{x}<br>Count: %{z}<extra></extra>",
                            showscale=False,
                        ))
                        fig_conf.update_layout(
                            title=dict(text="Agreement Matrix (TM vs LLM)", font=dict(size=13)),
                            height=320, margin=dict(l=80, r=20, t=40, b=40),
                            xaxis=dict(tickfont=dict(size=11)),
                            yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)

                # ── 3. Text Matching Verification Status Distribution ──
                st.markdown("##### Pkg 1: Text Matching Verification Status")
                _vs_counts = _cmp_df["tm_status"].value_counts()
                _vs_colors = {
                    "VERIFIED_M1": "#2E7D32", "VERIFIED_M2": "#F57F17",
                    "TEXT_MATCH_NO_CONTEXT": "#FFD54F", "KEYWORD_WITH_CONTEXT": "#FF8F00",
                    "KEYWORD_ONLY": "#FFD54F", "NO_MATCH": "#B71C1C",
                    "NO_FILE": "#9E9E9E", "NO_MISSION": "#B71C1C", "NO_DATA": "#BDBDBD",
                }
                if len(_vs_counts) > 0:
                    fig_vs = go.Figure(data=[go.Bar(
                        x=_vs_counts.index.tolist(),
                        y=_vs_counts.values.tolist(),
                        marker_color=[_vs_colors.get(s, "#999") for s in _vs_counts.index],
                        text=_vs_counts.values.tolist(),
                        textposition="auto",
                    )])
                    fig_vs.update_layout(
                        height=320, margin=dict(l=40, r=20, t=20, b=80),
                        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
                        yaxis=dict(title="Companies"),
                    )
                    st.plotly_chart(fig_vs, use_container_width=True)

                # ── 4. LLM Extraction vs Text Matching: Word Overlap ──
                _has_llm_ext = _cmp_df[_cmp_df["llm_ext_verdict"] != "N/A"]
                if not _has_llm_ext.empty:
                    st.markdown("##### Pkg 2 LLM Extraction vs Pkg 1 Text Matching")
                    st.caption("Word overlap between missions extracted by Pkg 1 (text matching) vs Pkg 2 (full LLM extraction).")

                    _ov1, _ov2 = st.columns(2)

                    with _ov1:
                        # Overlap histogram
                        _overlap_vals = _has_llm_ext["word_overlap"].tolist()
                        fig_hist = go.Figure(data=[go.Histogram(
                            x=_overlap_vals, nbinsx=20,
                            marker_color="#1565C0",
                            hovertemplate="Overlap: %{x:.2f}<br>Count: %{y}<extra></extra>",
                        )])
                        fig_hist.update_layout(
                            title=dict(text="Word Overlap Distribution", font=dict(size=13)),
                            height=320, margin=dict(l=40, r=20, t=40, b=40),
                            xaxis=dict(title="Jaccard Word Overlap", range=[0, 1]),
                            yaxis=dict(title="Companies"),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)

                    with _ov2:
                        # LLM extraction source type distribution
                        _ext_src_counts = _has_llm_ext["llm_ext_src"].value_counts()
                        _ext_src_colors = {
                            "EXPLICIT": "#2E7D32", "IMPLIED": "#F57F17",
                            "DERIVED": "#E65100", "NONE": "#B71C1C", "N/A": "#9E9E9E",
                        }
                        if len(_ext_src_counts) > 0:
                            fig_src = go.Figure(data=[go.Pie(
                                labels=_ext_src_counts.index.tolist(),
                                values=_ext_src_counts.values.tolist(),
                                marker=dict(colors=[_ext_src_colors.get(s, "#999") for s in _ext_src_counts.index]),
                                hole=0.4,
                                textinfo="label+value+percent",
                            )])
                            fig_src.update_layout(
                                title=dict(text="LLM Extraction Source Type", font=dict(size=13)),
                                height=320, margin=dict(l=20, r=20, t=40, b=20),
                                showlegend=False,
                            )
                            st.plotly_chart(fig_src, use_container_width=True)

                    # Agreement summary metrics
                    _high_agree = len(_has_llm_ext[_has_llm_ext["word_overlap"] > 0.5])
                    _low_agree = len(_has_llm_ext[_has_llm_ext["word_overlap"] < 0.2])
                    _total_ext = len(_has_llm_ext)

                    _m1, _m2, _m3, _m4 = st.columns(4)
                    with _m1:
                        st.metric("Total Compared", _total_ext)
                    with _m2:
                        st.metric("High Agreement (>50%)", _high_agree,
                                  delta=f"{_high_agree/_total_ext*100:.0f}%" if _total_ext else "0%")
                    with _m3:
                        st.metric("Low Agreement (<20%)", _low_agree,
                                  delta=f"{_low_agree/_total_ext*100:.0f}%" if _total_ext else "0%",
                                  delta_color="inverse")
                    with _m4:
                        _avg_overlap = _has_llm_ext["word_overlap"].mean()
                        st.metric("Avg Overlap", f"{_avg_overlap:.1%}")

                # ── 5. Per-company comparison table ──
                st.markdown("##### Per-Company Comparison")
                _tbl_cols = ["name", "tm_status", "tm_quality", "llm_verdict", "llm_type", "llm_quality", "llm_conf"]
                _tbl_rename = {
                    "name": "Company", "tm_status": "Pkg1 Status", "tm_quality": "Pkg1 Quality",
                    "llm_verdict": "Pkg1 LLM Verdict", "llm_type": "Pkg1 LLM Type",
                    "llm_quality": "Pkg1 LLM Quality", "llm_conf": "Pkg1 LLM Conf%",
                }
                if not _has_llm_ext.empty:
                    _tbl_cols += ["llm_ext_src", "llm_ext_quality", "word_overlap"]
                    _tbl_rename.update({
                        "llm_ext_src": "Pkg2 Source", "llm_ext_quality": "Pkg2 Quality",
                        "word_overlap": "Pkg1/Pkg2 Overlap",
                    })
                _tbl = _cmp_df[[c for c in _tbl_cols if c in _cmp_df.columns]].rename(columns=_tbl_rename)

                def _color_verdict(val):
                    colors = {"YES": "background-color: #C8E6C9", "PARTIAL": "background-color: #FFF9C4",
                              "NO": "background-color: #FFCDD2", "N/A": "background-color: #F5F5F5"}
                    return colors.get(val, "")

                with st.expander("View Full Comparison Table", expanded=False):
                    st.dataframe(
                        _tbl.style.applymap(_color_verdict, subset=["Pkg1 LLM Verdict"] if "Pkg1 LLM Verdict" in _tbl.columns else []),
                        use_container_width=True, hide_index=True,
                    )

                # CSV export
                _cmp_csv = _cmp_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Comparison Data (CSV)",
                    data=_cmp_csv,
                    file_name="tm_vs_llm_comparison.csv",
                    mime="text/csv",
                    key="dl_cmp_csv",
                )

        except Exception as _cmp_err:
            st.warning(f"Comparison analysis unavailable: {_cmp_err}")


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
    _is_pkg2_t4 = st.session_state.validation_approach != "Pkg 1: Text Matching"
    mission_kg, _ml_t4, _ = get_active_mission(company, _is_pkg2_t4)

    if mission_kg:
        tier_kg = overview_kg.get("mission_tier", 0)
        source_kg = overview_kg.get("mission_source", "unknown")
        _kg_schema_label = kg_mode.split("(")[-1].replace(")", "").strip() if "(" in kg_mode else "Full KG"
        st.markdown(
            f'<div style="background:#E8F5E9;border:1px solid #C8E6C9;border-radius:8px;padding:14px;margin-bottom:12px;">'
            f'<strong>Mission Statement:</strong> <em>"{mission_kg}"</em><br>'
            f'<span style="font-size:0.8rem;color:#555;">'
            f'Source: {source_kg.upper()} | Tier: {tier_kg} | Schema: {_kg_schema_label}</span>'
            f'</div>',
            unsafe_allow_html=True)

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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5: VERIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("MSG-KG v4.0 — Mission Statement Intelligence | "
           "Data: SEC 10-K Filings | Finance Research")
