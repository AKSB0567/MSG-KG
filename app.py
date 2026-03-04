import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSG-KG v2.0 — KG-RAG Portfolio Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# MSG-KG v2.0 — KG-RAG Portfolio Intelligence Interface
# Streamlit application with 6 tabs: Overview, Ask KG-RAG, Comparison,
# Evidence Explorer, Mission→HCM Analysis, KG View.
#
# v2.0 Changes:
# - New companies (AMD, ALX, LNG) with KG-derived data
# - Removed Good/Bad classification → score-based analysis only
# - Added Mission→HCM Analysis tab
# - Uses rdflib-based TTL KG loading via rag_engine
# - Switched from FinBERT to all-MiniLM-L6-v2 embeddings

import os, re, json, pathlib, sys
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
ITEM1_DIR  = BASE_DIR / "data" / "item1"
KG_DIR     = BASE_DIR / "MSGKG"

# ── Custom CSS Loader ────────────────────────────────────────────────────
def load_css(file_name):
    """Injects custom CSS from a local file into the Streamlit app."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = BASE_DIR / "assets" / "custom.css"
if css_path.exists():
    load_css(str(css_path))

# ── Company Metadata ────────────────────────────────────────────────────
# Score-based only — no Good/Bad classification
COMPANIES = {
    "AMD": {
        "name": "Advanced Micro Devices, Inc.",
        "sector": "Semiconductors",
        "fiscal_year": "FY2024",
        "cik": "0000002488",
        "filing_id": "0000002488-25-000012",
        "data_file": "012.txt",
        "ttl_file": "AMD.ttl",
        "sec_link": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000002488&type=10-K",
        "mission": "To build great products that accelerate next-generation computing experiences through high-performance and adaptive computing technology.",
    },
    "ALX": {
        "name": "Alexander & Baldwin, Inc.",
        "sector": "Real Estate / Diversified",
        "fiscal_year": "FY2024",
        "cik": "0000003499",
        "filing_id": "0000003499-25-000004",
        "data_file": "004.txt",
        "ttl_file": "ALX.ttl",
        "sec_link": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000003499&type=10-K",
        "mission": "To be Hawaii's premier commercial real estate company, enriching the lives of our team members, the communities we serve, and our shareholders.",
    },
    "LNG": {
        "name": "Cheniere Energy, Inc.",
        "sector": "Energy / LNG",
        "fiscal_year": "FY2024",
        "cik": "0000003570",
        "filing_id": "0000003570-25-000033",
        "data_file": "033.txt",
        "ttl_file": "LNG.ttl",
        "sec_link": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000003570&type=10-K",
        "mission": "To be the world's leading full-service LNG provider, delivering clean, reliable, and affordable energy to the global market.",
    },
}

# ── Alignment scoring dimensions ─────────────────────────────────────────
# Each dimension scores on 0–3 scale:
#   0 = Not present, 1 = Weak, 2 = Moderate, 3 = Strong
ALIGNMENT_DIMENSIONS = [
    "Mission Clarity",
    "Vision → Strategy Linkage",
    "Strategy → Operations Grounding",
    "Operations → Financial Linkage",
    "HCM Disclosure Depth",
    "Risk Awareness & Mitigation",
    "Initiative Specificity",
    "Capability Articulation",
]


def _compute_alignment_scores(ticker: str, overview: dict) -> dict:
    """Compute alignment scores from KG-derived company overview data."""
    scores = {}
    
    # Mission Clarity: based on whether mission is present and descriptive
    mission = overview.get("mission", "")
    scores["Mission Clarity"] = 3 if len(str(mission)) > 20 else (1 if mission else 0)
    
    # Vision → Strategy Linkage: based on number of objectives
    n_obj = len(overview.get("objectives", []))
    scores["Vision → Strategy Linkage"] = min(3, n_obj)
    
    # Strategy → Operations Grounding: capabilities + initiatives
    n_cap = len(overview.get("capabilities", []))
    n_init = len(overview.get("initiatives", []))
    scores["Strategy → Operations Grounding"] = min(3, n_cap + n_init)
    
    # Operations → Financial Linkage: financial metrics
    n_fin = len(overview.get("financial_metrics", []))
    scores["Operations → Financial Linkage"] = 3 if n_fin >= 50 else (2 if n_fin >= 10 else (1 if n_fin > 0 else 0))
    
    # HCM Disclosure Depth: HCM-related nodes
    n_hcm = len(overview.get("hcm_metrics", []))
    scores["HCM Disclosure Depth"] = 3 if n_hcm >= 3 else (2 if n_hcm >= 1 else 1)
    
    # Risk Awareness & Mitigation: risk themes
    n_risk = len(overview.get("risks", []))
    scores["Risk Awareness & Mitigation"] = 3 if n_risk >= 10 else (2 if n_risk >= 3 else (1 if n_risk > 0 else 0))
    
    # Initiative Specificity
    scores["Initiative Specificity"] = 3 if n_init >= 5 else (2 if n_init >= 2 else (1 if n_init > 0 else 0))
    
    # Capability Articulation
    scores["Capability Articulation"] = 3 if n_cap >= 3 else (2 if n_cap >= 1 else 1)
    
    return scores


# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 50%, #3a7ab5 100%);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Pipeline bar */
    .pipeline-bar {
        background: linear-gradient(90deg, #e8f0fe, #d2e3fc);
        border: 1px solid #c0d7f0;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.2rem;
        font-family: monospace;
    }
    .pipeline-bar .pipeline-title {
        font-weight: 700;
        font-size: 1rem;
        color: #1a3a5c;
    }
    .pipeline-bar .pipeline-steps {
        color: #2962a8;
        font-size: 0.9rem;
        font-weight: 600;
    }

    /* Result cards */
    .result-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .result-card h3 {
        margin-top: 0;
        color: #1a3a5c;
        font-size: 1.1rem;
        border-bottom: 2px solid #2962a8;
        padding-bottom: 0.4rem;
    }

    /* Evidence span */
    .evidence-item {
        background: #ffffff;
        border-left: 3px solid #2962a8;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        border-radius: 0 4px 4px 0;
    }

    /* KG path */
    .kg-path {
        font-family: 'Consolas', 'Courier New', monospace;
        background: #f0f4f8;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        font-size: 0.88rem;
        border-left: 3px solid #4a9eff;
    }
    .kg-path-bold {
        font-weight: 700;
        color: #1a3a5c;
    }

    /* Interactive button styles */
    .stButton>button {
        border-radius: 6px;
        background-color: #2962a8;
        color: white;
        border: none;
        padding: 0.4rem 1.2rem;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1a3a5c;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        color: white;
    }

    /* Metric Card */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f5);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e4e8;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Panel header */
    .panel-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3a5f;
        padding: 0.6rem 1rem;
        background: linear-gradient(90deg, #e8f0fe, #f8fafc);
        border-radius: 6px;
        border-left: 4px solid #2962a8;
        margin: 1rem 0;
    }

    /* Sector badge */
    .sector-badge {
        display: inline-block;
        background: linear-gradient(135deg, #2962a8, #1a3a5c);
        color: white;
        padding: 4px 14px;
        border-radius: 14px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Score bar */
    .score-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        height: 12px;
        margin: 4px 0;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
    }

    /* Tab styling fix */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }

    /* Comparison table */
    .compare-table th {
        background: #2962a8;
        color: white;
        padding: 0.6rem;
    }
    .compare-table td {
        padding: 0.6rem;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ── Display text cleanup utilities ───────────────────────────────────────

def clean_kg_node_name(name: str) -> str:
    """Convert KG node names like AMD_AdaptiveComputingLeadership to readable text."""
    if not name:
        return ""
    # Remove company prefix (e.g., AMD_, LNG_, ALX_)
    for prefix in ["AMD_", "ALX_", "LNG_", "UNM_", "AAL_", "WRB_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Split camelCase and underscores
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = name.replace('_', ' ').replace('-', ' ')
    return name.strip().title()


def clean_display_text(text: str) -> str:
    """Clean KG-extracted text for display."""
    if not text or not text.strip():
        return ""
    t = " ".join(text.split())
    t = t.lstrip("•·-– ")
    t = re.sub(r'^\|?\s*\d{4}\s+Form\s+10-K\s*\|?\s*\d*\s*', '', t).strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in '.!?:"\'':
        t = t.rstrip(',;') + '.'
    return t


def format_risk_name(raw_name: str) -> str:
    """Format risk theme names for display."""
    return clean_kg_node_name(raw_name)


def format_capability(name: str) -> str:
    """Capitalize capability names for display."""
    return clean_kg_node_name(name)


# ── Helper: safe import of rag_engine ────────────────────────────────────
@st.cache_resource
def load_rag_engine():
    """Import and initialize RAG engine on first call."""
    try:
        import rag_engine
        return rag_engine
    except ImportError as e:
        st.warning(f"RAG engine not available: {e}")
        return None


def load_chunks_df(ticker: str) -> pd.DataFrame:
    """Load chunks for a company as a DataFrame."""
    chunk_path = CHUNKS_DIR / f"{ticker}_chunks.json"
    if chunk_path.exists():
        with open(chunk_path, encoding="utf-8") as f:
            chunks = json.load(f)
        return pd.DataFrame(chunks)
    return pd.DataFrame()


@st.cache_resource
def load_kg_graph() -> nx.DiGraph:
    """Load KG from rag_engine (TTL-based) or fallback to CSV."""
    rag = load_rag_engine()
    if rag:
        try:
            return rag.load_kg()
        except Exception:
            pass
    # Fallback to CSV
    G = nx.DiGraph()
    nodes_path = KG_DIR / "nodes.csv"
    edges_path = KG_DIR / "edges.csv"
    if nodes_path.exists():
        df = pd.read_csv(nodes_path, encoding="utf-8")
        for _, row in df.iterrows():
            attrs = {k: v for k, v in row.items()
                     if k != ":ID" and pd.notna(v) and str(v).strip()}
            G.add_node(row[":ID"], **attrs)
    if edges_path.exists():
        df = pd.read_csv(edges_path, encoding="utf-8")
        for _, row in df.iterrows():
            G.add_edge(row[":START_ID"], row[":END_ID"], relation=row[":TYPE"])
    return G


@st.cache_data
def get_company_overview_cached(ticker: str) -> dict:
    """Get company overview from rag_engine, cached."""
    rag = load_rag_engine()
    if rag:
        try:
            return rag.get_company_overview(ticker)
        except Exception:
            pass
    return {"ticker": ticker, "name": ticker, "mission": "",
            "objectives": [], "capabilities": [], "initiatives": [],
            "risks": [], "hcm_metrics": [], "financial_metrics": [],
            "kg_stats": {"nodes": 0, "edges": 0}}


# ── Header ───────────────────────────────────────────────────────────────
def render_header():
    """Renders the global dashboard header."""
    st.markdown("""
    <div style="background-color:#CD1515; padding:15px 20px; border-bottom:2px solid #AA8F00; margin-bottom:20px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 2px 5px rgba(0,0,0,0.15);">
        <div style="display:flex; align-items:center;">
             <div style="font-size:26px; font-weight:700; color:#FFFFFF; margin-right:15px; letter-spacing:-0.5px;">
                 MSG-KG
             </div>
             <div style="border-left:1px solid #FFFFFF50; padding-left:15px; font-size:16px; font-weight:500; color:#FFFFFFee;">
                 Portfolio Intelligence System
             </div>
        </div>
        <div style="text-align:right;">
             <div style="font-size:12px; font-weight:600; color:#FFFFFFcc; letter-spacing:0.5px; text-transform:uppercase;">
                 Enterprise Edition v2.0
             </div>
             <div style="font-size:11px; color:#FFFFFFaa; margin-top:2px;">
                 Secure Connection <span style="color:#FFF;">●</span>
             </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Common Header ────────────────────────────────────────────────────────
render_header()

# ── MAIN LAYOUT CONSTRUCTION ────────────────────────────────────────────
col_filters, col_content = st.columns([1, 4])

# ── Left Column: Contextual Filters ─────────────────────────────────────
with col_filters:
    st.markdown("### 🔍 Filters")
    st.markdown("---")

    # Sector filter (replaces old Good/Bad filter)
    st.caption("SECTOR FILTER")
    sectors = sorted(set(m["sector"] for m in COMPANIES.values()))
    selected_sectors = st.multiselect("Sectors", ["All"] + sectors,
                                       default=["All"], label_visibility="collapsed")

    if "All" in selected_sectors or not selected_sectors:
        available_tickers = list(COMPANIES.keys())
    else:
        available_tickers = [t for t, m in COMPANIES.items()
                             if m["sector"] in selected_sectors]

    # Company
    st.caption("SELECT COMPANY")
    company = st.selectbox("Company", available_tickers,
                           format_func=lambda t: f"{t} — {COMPANIES[t]['name']}",
                           label_visibility="collapsed")

    fiscal_year = COMPANIES[company]["fiscal_year"]
    sector = COMPANIES[company]["sector"]
    st.info(f"📅 {fiscal_year}  ·  🏭 {sector}")

    st.markdown("---")

    # Evidence Sources
    st.caption("DATA SOURCES")
    evidence_sources = st.multiselect("Sources", ["10-K", "Earnings", "News"],
                                      default=["10-K"], label_visibility="collapsed")

    st.markdown("---")

    # Model Params
    with st.expander("⚙️ RAG Model Config", expanded=False):
        top_k_vector = st.slider("chunks (Vec)", 1, 50, 10)
        rerank_val = st.toggle("Rerank", value=True)
        rerank = "ON" if rerank_val else "OFF"
        st.markdown("---")
        top_k_graph = st.slider("nodes (KG)", 1, 50, 10)
        explain_val = st.toggle("Explain", value=True)
        explain_mode = "ON" if explain_val else "OFF"


# ── Right Column: Main Content (Tabs) ────────────────────────────────────
with col_content:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🤖 Ask KG-RAG",
        "⚖️ Competitive Landscape",
        "🔍 Evidence Explorer",
        "👥 Mission → HCM Analysis",
        "🕸️ KG View",
    ])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    meta = COMPANIES[company]
    st.markdown(f"### Company Overview: **{company}** — {meta['name']}")

    st.info(
        "📖 **About this tab:** This overview is generated from the company's SEC 10-K filing "
        "(Item 1 — Business) combined with knowledge graph (KG) data extracted using our ontology. "
        f"Data sources: `kg1.ttl` (shared ontology) + `{meta.get('ttl_file', 'N/A')}` (company-specific KG) "
        f"· CIK: `{meta.get('cik', 'N/A')}` · Filing: `{meta.get('filing_id', 'N/A')}` "
        "· All metrics below are **automatically derived** from the Knowledge Graph — not manually curated."
    )

    overview = get_company_overview_cached(company)
    n_risks = len(overview.get("risks", []))
    n_fin = len(overview.get("financial_metrics", []))
    n_init = len(overview.get("initiatives", []))
    n_obj = len(overview.get("objectives", []))
    n_cap = len(overview.get("capabilities", []))
    n_hcm = len(overview.get("hcm_metrics", []))

    # ── Top Metrics Row ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("🏢 Ticker", company)
    with m2:
        st.metric("📅 Fiscal Year", meta["fiscal_year"])
    with m3:
        st.metric("🔗 KG Nodes", overview["kg_stats"]["nodes"])
    with m4:
        st.metric("➡️ KG Edges", overview["kg_stats"]["edges"])
    with m5:
        st.metric("⚠️ Risk Themes", n_risks)
    with m6:
        st.metric("💰 Financial Metrics", n_fin)

    st.markdown("---")

    # ── Data Source & Filing Panel ──
    with st.expander("📦 Data Sources & Filing Details", expanded=True):
        ds1, ds2 = st.columns(2)
        with ds1:
            st.markdown(f"""
            **SEC Filing Information**
            | Field | Value |
            |-------|-------|
            | **CIK** | `{meta.get('cik', 'N/A')}` |
            | **Filing ID** | `{meta.get('filing_id', 'N/A')}` |
            | **Data File** | `cleaned_10-K_{meta.get('filing_id', 'N/A')}.txt` |
            | **SEC EDGAR** | [View on SEC]({meta.get('sec_link', '#')}) |
            """)
        with ds2:
            st.markdown(f"""
            **Knowledge Graph Sources**
            | Source | Description |
            |--------|-------------|
            | `kg1.ttl` | Shared ontology instances (Mission, Strategy, Capability) |
            | `{meta.get('ttl_file', 'N/A')}` | Company-specific KG (Risks, Financials, Initiatives) |
            | `schema_1.owl` | Ontology schema definition |

            **RAG Pipeline**: MiniLM-L6-v2 → FAISS → Cross-Encoder Reranker → Mixtral-8x7B
            """)
        # Show chunk info
        chunk_path = CHUNKS_DIR / f"{company}_chunks.json"
        item1_path = ITEM1_DIR / f"{company}_item1.txt"
        if chunk_path.exists():
            with open(chunk_path, encoding="utf-8") as f:
                chunks_data = json.load(f)
            total_words = sum(c.get("end_word", 0) - c.get("start_word", 0) for c in chunks_data)
            sections = set(c.get("section", "Unknown") for c in chunks_data)
            st.success(f"📄 **{len(chunks_data)}** chunks · ~**{total_words:,}** words · "
                      f"**{len(sections)}** sections: {', '.join(sorted(sections))}")
        elif item1_path.exists():
            text = item1_path.read_text(encoding="utf-8")
            st.success(f"📄 Item 1 text loaded: **{len(text):,}** characters")

    st.markdown("---")

    # ── Main Content: Mission + Objectives + Capabilities ──
    col_l, col_r = st.columns(2)

    with col_l:
        # ── Mission Card ──
        curated_mission = meta.get("mission", "")
        kg_mission = overview.get("mission", "")
        display_mission = curated_mission if curated_mission else clean_kg_node_name(kg_mission)
        sector_text = meta.get("sector", "")

        st.markdown(f"""
        <div class="result-card" style="border-top:4px solid #2962a8;">
            <h4 style="margin-top:0;color:#1a3a5c;">🎯 Mission Statement</h4>
            <div style="background:linear-gradient(135deg,#f0f4f8,#e8eef5);padding:1rem 1.2rem;
                        border-radius:8px;margin:0.5rem 0;font-size:1.05rem;line-height:1.6;
                        border-left:4px solid #2962a8;">
                <em>"{display_mission}"</em>
            </div>
            <div style="margin-top:0.6rem;">
                <span class="sector-badge">🏭 {sector_text}</span>
                <span style="margin-left:10px;font-size:0.82rem;color:#666;">CIK: {meta.get('cik','N/A')}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Strategic Objectives ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"#### 📋 Strategic Objectives ({n_obj})")
        if overview["objectives"]:
            for obj in overview["objectives"]:
                cleaned = clean_kg_node_name(obj)
                if cleaned:
                    st.markdown(f"- {cleaned}")
        else:
            st.caption("No objectives extracted from KG.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        # ── Capabilities ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"#### 🛠️ Capabilities ({n_cap})")
        if overview["capabilities"]:
            for cap in overview["capabilities"]:
                st.markdown(f"- {format_capability(cap)}")
        else:
            st.caption("No capabilities extracted from KG.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── HCM Metrics ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"#### 👥 Human Capital Metrics ({n_hcm})")
        if overview["hcm_metrics"]:
            for hcm in overview["hcm_metrics"]:
                st.markdown(f"- {clean_kg_node_name(hcm)}")
        else:
            st.caption("No HCM metrics found in KG.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Risk Themes (full-width with columns) ──
    st.markdown("---")
    st.markdown(f'<div class="panel-header">⚠️ Risk Themes ({n_risks} identified)</div>', unsafe_allow_html=True)
    risks = overview.get("risks", [])
    if risks:
        risk_cols = st.columns(3)
        for i, risk in enumerate(risks):
            with risk_cols[i % 3]:
                st.markdown(f"- {format_risk_name(risk)}")
    else:
        st.caption("No risk themes extracted from KG.")

    # ── Initiatives (full-width) ──
    initiatives = overview.get("initiatives", [])
    if initiatives:
        st.markdown("---")
        st.markdown(f'<div class="panel-header">🚀 Initiatives ({n_init})</div>', unsafe_allow_html=True)
        init_cols = st.columns(2)
        for i, init in enumerate(initiatives):
            cleaned = clean_kg_node_name(init)
            if cleaned:
                with init_cols[i % 2]:
                    st.markdown(f"- {cleaned}")

    # ── Financial Metrics Summary ──
    fin_metrics = overview.get("financial_metrics", [])
    if fin_metrics:
        st.markdown("---")
        with st.expander(f"💰 Financial Metrics ({n_fin} from KG)", expanded=False):
            fin_cols = st.columns(3)
            for i, fm in enumerate(fin_metrics):
                with fin_cols[i % 3]:
                    st.markdown(f"- {clean_kg_node_name(fm)}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Ask KG-RAG
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.info(
        "📖 **About this tab:** Ask natural-language questions about any company's 10-K filing. "
        "The system uses a **hybrid Graph-RAG pipeline**: "
        "(1) **Vector Retrieval** — MiniLM-L6-v2 embeddings + FAISS index search over 10-K chunks, "
        "(2) **Graph Retrieval** — entity linking + KG subgraph extraction from the TTL knowledge graph, "
        "(3) **Cross-Encoder Reranker** — reranks combined results by relevance, "
        "(4) **LLM Reasoning** — Mixtral-8x7B (HuggingFace Inference API) generates the final answer "
        "grounded in both textual evidence and KG structure."
    )

    # Pipeline visualization
    pipeline_steps = "Vector Retrieval → Graph Retrieval → Reranker → KG-RAG Reasoner"
    st.markdown(f"""
    <div class="pipeline-bar">
        <span class="pipeline-title">Retrieval Pipeline</span><br>
        <span class="pipeline-steps">{pipeline_steps}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Ask KG-RAG (Q/A)")

    question = st.text_input(
        "Enter your question:",
        value="What is the mission and what strategies support it?",
        key="qa_question",
    )

    if st.button("🔍 Ask", key="qa_ask", type="primary"):
        rag = load_rag_engine()

        if rag:
            config = {
                "top_k_vector": top_k_vector,
                "top_k_graph": top_k_graph,
                "rerank": rerank == "ON",
                "explain": explain_mode == "ON",
            }

            # Real-time progress timer
            import time as _time
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            t_start = _time.time()

            pipeline_stages = [
                ("🔎 Vector Retrieval (MiniLM-L6-v2 embedding + FAISS search)...", 0.15),
                ("🕸️ Graph Retrieval (KG subgraph extraction)...", 0.30),
                ("📊 Reranking (Cross-encoder scoring)...", 0.50),
                ("🤖 KG-RAG Reasoning (Mixtral-8x7B generation)...", 0.75),
            ]

            progress_bar = progress_placeholder.progress(0, text="Initializing pipeline...")
            for stage_text, stage_pct in pipeline_stages:
                elapsed = _time.time() - t_start
                progress_bar.progress(stage_pct, text=f"{stage_text}  ⏱️ {elapsed:.1f}s")
                if stage_pct < 0.50:
                    _time.sleep(0.3)

            # Actually run the pipeline
            result = rag.ask(question, company, config)
            elapsed_total = _time.time() - t_start
            progress_bar.progress(1.0, text=f"✅ Complete — {elapsed_total:.1f}s total")
            status_placeholder.success(f"Pipeline finished in **{elapsed_total:.1f} seconds** · "
                                       f"{len(result.evidence_spans)} evidence spans · "
                                       f"{len(result.kg_paths)} KG paths")

            # Model Answer + Evidence Spans
            col_answer, col_evidence = st.columns([3, 2])

            with col_answer:
                st.markdown("""<div class="result-card"><h3>Model Answer</h3>""",
                           unsafe_allow_html=True)
                st.markdown(result.answer)
                st.markdown("</div>", unsafe_allow_html=True)

            with col_evidence:
                st.markdown("""<div class="result-card"><h3>Retrieved Evidence Spans</h3>""",
                           unsafe_allow_html=True)
                for span in result.evidence_spans:
                    span_text_clean = clean_display_text(span.text) if span.text else ""
                    st.markdown(
                        f'<div class="evidence-item">'
                        f'<strong>{span.doc_type}</strong> | {span.section} | '
                        f'Page {span.page} | Conf {span.confidence}'
                        f'<br><span style="font-size:0.82rem;color:#444;">{span_text_clean[:200]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                if not result.evidence_spans:
                    st.caption("No evidence spans retrieved.")
                st.markdown("</div>", unsafe_allow_html=True)

            # KG Path Traces
            if explain_mode == "ON" and result.kg_paths:
                st.markdown("""<div class="result-card">
                    <h3>KG Path Traces</h3>
                    <p style="color:#666; font-size:0.85rem;">Paths used in reasoning (Explain mode)</p>
                """, unsafe_allow_html=True)
                for i, kp in enumerate(result.kg_paths[:8]):
                    weight = "kg-path-bold" if i == 0 else ""
                    st.markdown(
                        f'<div class="kg-path {weight}">{kp.path}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("RAG engine not loaded. Install dependencies and ensure data files exist.")

    else:
        # Placeholder
        col_answer, col_evidence = st.columns([3, 2])
        with col_answer:
            st.markdown("""<div class="result-card"><h3>Model Answer</h3>""",
                       unsafe_allow_html=True)
            st.caption("Enter a question and click 'Ask' to get started.")
            st.markdown("</div>", unsafe_allow_html=True)
        with col_evidence:
            st.markdown("""<div class="result-card"><h3>Retrieved Evidence Spans</h3>""",
                       unsafe_allow_html=True)
            st.caption("Evidence will appear here after asking a question.")
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: Comparison — Competitive Landscape Dashboard
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.info(
        "📖 **About this tab:** Compares companies across 8 alignment dimensions using scores "
        "**automatically derived** from KG node counts. Each dimension is scored 0–3 based on the "
        "richness of KG data: **Mission Clarity** (mission node presence), **Vision→Strategy Linkage** "
        "(number of strategic objectives), **Strategy→Operations Grounding** (capabilities + initiatives), "
        "**Operations→Financial Linkage** (financial metrics count), **HCM Disclosure Depth** (HCM nodes), "
        "**Risk Awareness** (risk themes), **Initiative Specificity** (initiative count), and "
        "**Capability Articulation** (capability nodes). Higher scores indicate deeper disclosure quality."
    )

    # ── 1. Portfolio Heatmap Section ──────────────────────────────────
    st.markdown('<div class="panel-header">1. Portfolio Alignment Heatmap</div>', unsafe_allow_html=True)
    st.caption("Score intensity (0-3): **0**=Missing · **1**=Weak · **2**=Moderate · **3**=Strong")

    # Compute scores for all companies
    all_scores = {}
    for t in COMPANIES:
        ov = get_company_overview_cached(t)
        all_scores[t] = _compute_alignment_scores(t, ov)

    # Heatmap Controls
    h_col1, h_col2, h_col3 = st.columns([1.2, 1.5, 0.8])
    with h_col1:
        hm_sort = st.selectbox("Sort Heatmap by:", ["Total Score (High → Low)", "Total Score (Low → High)", "Ticker (A-Z)"])
    with h_col2:
        hm_dims = st.multiselect("Select Dimensions:", ALIGNMENT_DIMENSIONS, default=ALIGNMENT_DIMENSIONS)
    with h_col3:
        st.write("")
        fit_width = st.checkbox("Fit to Width", value=True)

    if not hm_dims:
        st.warning("Please select at least one dimension.")
        st.stop()

    # Build Heatmap Data
    heat_rows = []
    for t in sorted(COMPANIES.keys()):
        meta_t = COMPANIES[t]
        scores = all_scores.get(t, {})
        row = {"Ticker": t, "Sector": meta_t.get("sector", "")}
        row["Total"] = sum(scores.get(d, 0) for d in hm_dims)
        for dim in hm_dims:
            row[dim] = scores.get(dim, 0)
        heat_rows.append(row)

    if heat_rows:
        df_heat = pd.DataFrame(heat_rows)
        if "High → Low" in hm_sort:
            df_heat = df_heat.sort_values("Total", ascending=False)
        elif "Low → High" in hm_sort:
            df_heat = df_heat.sort_values("Total", ascending=True)
        else:
            df_heat = df_heat.sort_values("Ticker", ascending=True)

        df_display = df_heat.set_index("Ticker").drop(columns=["Sector", "Total"])

        # Render Interactive Heatmap using Plotly
        import plotly.graph_objects as go

        hover_text = []
        for index, row in df_display.iterrows():
            hover_text.append([f"Ticker: {index}<br>Dimension: {col}<br>Score: {val}" for col, val in row.items()])

        cell_height = 50
        cell_width = 130
        n_rows = len(df_display)
        n_cols = len(df_display.columns)
        fig_height = max(350, n_rows * cell_height + 120)
        fig_width = n_cols * cell_width + 150

        fig = go.Figure(data=go.Heatmap(
            z=df_display.values,
            x=df_display.columns,
            y=df_display.index,
            text=df_display.values,
            texttemplate="%{text}",
            colorscale='Viridis',
            zmin=0, zmax=3,
            hoverinfo='text',
            hovertext=hover_text,
            xgap=2, ygap=2
        ))

        fig.update_layout(
            title_text=None,
            height=fig_height,
            xaxis=dict(side="bottom", tickangle=-45, tickmode='linear'),
            yaxis=dict(autorange="reversed", tickmode='linear'),
            margin=dict(l=10, r=10, t=10, b=10),
        )

        if fit_width:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            fig.update_layout(width=fig_width)
            st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': True})

        # Total score summary
        st.markdown("#### 📊 Total Alignment Scores")
        score_cols = st.columns(len(COMPANIES))
        for i, (t, scores) in enumerate(sorted(all_scores.items(),
                                                key=lambda x: sum(x[1].values()),
                                                reverse=True)):
            total = sum(scores.values())
            max_possible = len(ALIGNMENT_DIMENSIONS) * 3
            pct = round(total / max_possible * 100) if max_possible else 0
            color = "#27ae60" if pct >= 70 else ("#f39c12" if pct >= 40 else "#e74c3c")
            with score_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{color};">{pct}%</div>
                    <div class="metric-label">{t}</div>
                    <div style="font-size:0.75rem;color:#888;margin-top:4px;">{total}/{max_possible}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No companies match the selected filter.")

    st.markdown("---")

    # ── 2. Head-to-Head Comparison ────────────────────────────────────
    st.markdown('<div class="panel-header">2. Head-to-Head Comparison</div>', unsafe_allow_html=True)
    st.caption("Select two companies to compare their Alignment Checklist and Evidence Chain.")

    compare_tickers = sorted(COMPANIES.keys())
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        company1 = st.selectbox("Company 1", compare_tickers, index=0, key="cmp1",
                                format_func=lambda t: f"{t} — {COMPANIES[t]['name']}")
    with col_c2:
        default_idx = min(1, len(compare_tickers) - 1)
        company2 = st.selectbox("Company 2", compare_tickers, index=default_idx, key="cmp2",
                                format_func=lambda t: f"{t} — {COMPANIES[t]['name']}")

    meta1 = COMPANIES[company1]
    meta2 = COMPANIES[company2]
    scores1 = all_scores.get(company1, {})
    scores2 = all_scores.get(company2, {})

    # Header cards (score-based colors)
    hdr1, hdr2 = st.columns(2)
    total1 = sum(scores1.values())
    total2 = sum(scores2.values())
    max_score = len(ALIGNMENT_DIMENSIONS) * 3
    pct1 = round(total1 / max_score * 100) if max_score else 0
    pct2 = round(total2 / max_score * 100) if max_score else 0
    color1 = "#27ae60" if pct1 >= 70 else ("#f39c12" if pct1 >= 40 else "#e74c3c")
    color2 = "#27ae60" if pct2 >= 70 else ("#f39c12" if pct2 >= 40 else "#e74c3c")

    with hdr1:
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {color1};">
            <h3 style="border:none;padding:0;">{company1} — {meta1['name']}</h3>
            <span class="sector-badge">🏭 {meta1.get('sector','')}</span>
            <p style="margin-top:0.6rem;font-size:0.88rem;color:#555;">
                📅 {meta1['fiscal_year']} &nbsp;|&nbsp;
                🎯 <em>{meta1.get('mission','')[:100]}</em>
            </p>
        </div>""", unsafe_allow_html=True)
    with hdr2:
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {color2};">
            <h3 style="border:none;padding:0;">{company2} — {meta2['name']}</h3>
            <span class="sector-badge">🏭 {meta2.get('sector','')}</span>
            <p style="margin-top:0.6rem;font-size:0.88rem;color:#555;">
                📅 {meta2['fiscal_year']} &nbsp;|&nbsp;
                🎯 <em>{meta2.get('mission','')[:100]}</em>
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Overall Alignment Score ──
    ov1, ov2, ov3 = st.columns([2, 1, 2])
    with ov1:
        st.metric(f"🏆 {company1} Alignment", f"{pct1}%", f"{total1}/{max_score}")
    with ov2:
        winner = company1 if total1 > total2 else (company2 if total2 > total1 else "Tie")
        winner_color = "#27ae60" if winner != "Tie" else "#888"
        st.markdown(f"""<div style="text-align:center;padding-top:0.5rem;">
            <span style="font-size:1.8rem;">⚔️</span><br>
            <span style="color:{winner_color};font-weight:700;font-size:1rem;">{winner} wins</span>
        </div>""", unsafe_allow_html=True)
    with ov3:
        st.metric(f"🏆 {company2} Alignment", f"{pct2}%", f"{total2}/{max_score}")

    st.markdown("---")

    # ── Dimension-by-dimension scorecard ──
    st.markdown("#### 📋 Alignment Scorecard — Dimension Checklist")
    st.caption("Each dimension scored 0–3: 0 = Not present · 1 = Weak · 2 = Moderate · 3 = Strong")

    score_icons = {0: "⬜", 1: "🟡", 2: "🟠", 3: "🟢"}
    table_data = []
    for dim in ALIGNMENT_DIMENSIONS:
        s1 = scores1.get(dim, 0)
        s2 = scores2.get(dim, 0)
        if s1 > s2:
            winner_icon = f"← {company1}"
        elif s2 > s1:
            winner_icon = f"{company2} →"
        else:
            winner_icon = "="
        table_data.append({
            "Dimension": dim,
            f"{company1}": f"{score_icons[s1]} {s1}/3",
            f"{company2}": f"{score_icons[s2]} {s2}/3",
            "Better": winner_icon,
        })

    df_scores = pd.DataFrame(table_data)
    st.dataframe(df_scores, use_container_width=True, hide_index=True,
                 column_config={
                     "Dimension": st.column_config.TextColumn("📊 Dimension", width=250),
                     f"{company1}": st.column_config.TextColumn(f"🏢 {company1}", width=120),
                     f"{company2}": st.column_config.TextColumn(f"🏢 {company2}", width=120),
                     "Better": st.column_config.TextColumn("🏆 Winner", width=120),
                 })

    st.markdown("---")

    # ── Evidence chain comparison ──
    ov1 = get_company_overview_cached(company1)
    ov2 = get_company_overview_cached(company2)

    ep1, ep2 = st.columns(2)

    def render_evidence_path(col, ticker, ov_data, meta_data):
        with col:
            total_s = sum(all_scores.get(ticker, {}).values())
            pct_s = round(total_s / max_score * 100) if max_score else 0
            ev_color = "#27ae60" if pct_s >= 70 else ("#f39c12" if pct_s >= 40 else "#e74c3c")
            st.markdown(f"**{ticker}** — {meta_data['name']}")

            steps = [
                ("🎯 Mission/Vision", meta_data.get("mission", "")[:120]),
                ("📋 Strategic Goals", " · ".join(clean_kg_node_name(o) for o in ov_data.get("objectives", [])[:3]) or "—"),
                ("🛠️ Capabilities", " · ".join(format_capability(c) for c in ov_data.get("capabilities", [])[:4]) or "—"),
                ("🚀 Initiatives", " · ".join(clean_kg_node_name(i) for i in ov_data.get("initiatives", [])[:3]) or "—"),
                ("⚠️ Risk Mitigation", " · ".join(format_risk_name(r) for r in ov_data.get("risks", [])[:3]) or "—"),
            ]

            for label, val in steps:
                st.markdown(f"""
                <div class="evidence-item" style="border-left-color:{ev_color};">
                    <strong>{label}</strong><br>
                    <span style="color:#555;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    render_evidence_path(ep1, company1, ov1, meta1)
    render_evidence_path(ep2, company2, ov2, meta2)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: Evidence Explorer
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### Evidence Explorer — {company}")

    st.info(
        "📖 **About this tab:** Browse the raw text chunks extracted from each company's SEC 10-K filing. "
        "Chunks are created using **section-aware sentence-level recursive chunking** — the 10-K text is first "
        "split by section headers (Business, Products, Competition, etc.), then into semantically coherent "
        "paragraphs of ~500 words each. These chunks serve as the retrieval corpus for the RAG pipeline. "
        "Use the keyword and section filters to explore specific topics."
    )

    df_chunks = load_chunks_df(company)

    if not df_chunks.empty:
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            keyword = st.text_input("🔎 Filter by keyword:", key="ev_keyword")
        with col_f2:
            if "section" in df_chunks.columns:
                sections = sorted(df_chunks["section"].dropna().unique())
                selected_sections = st.multiselect("📑 Filter by section:", sections,
                                                    default=sections, key="ev_sections")
            else:
                selected_sections = None

        # Apply filters
        filtered = df_chunks.copy()
        if keyword:
            filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]
        if selected_sections is not None and "section" in filtered.columns:
            filtered = filtered[filtered["section"].isin(selected_sections)]

        st.markdown(f"**Showing {len(filtered)} of {len(df_chunks)} chunks**")

        for _, row in filtered.iterrows():
            section_label = row.get("section", "")
            chunk_id = row.get("chunk_id", "")
            with st.expander(f"📄 {chunk_id} — Section: {section_label}"):
                st.markdown(row["text"])
                meta_parts = []
                if "start_word" in row and "end_word" in row:
                    meta_parts.append(f"Words {row['start_word']}–{row['end_word']}")
                if section_label:
                    meta_parts.append(f"Section: {section_label}")
                st.caption(" | ".join(meta_parts))
    else:
        st.warning(f"No chunks found for {company}. Run `python chunk_10k.py` first.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: Mission → HCM Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 👥 Mission → Human Capital Management Analysis")

    st.info(
        "📖 **About this tab:** Analyzes the connection between each company's **mission statement** and their "
        "**Human Capital Management (HCM)** disclosures as reported in the SEC 10-K filing. "
        "HCM metrics include workforce demographics, diversity & inclusion programs, compensation, "
        "training, safety, and employee engagement data. Companies with a strong mission→HCM linkage "
        "demonstrate that their stated purpose translates into concrete workforce practices. "
        "The **HCM Score** (0–3) is derived from the number of HCM-related nodes in the Knowledge Graph."
    )

    st.markdown("---")

    for ticker in sorted(COMPANIES.keys()):
        meta_t = COMPANIES[ticker]
        ov_t = get_company_overview_cached(ticker)
        scores_t = _compute_alignment_scores(ticker, ov_t)

        # HCM score color
        hcm_score = scores_t.get("HCM Disclosure Depth", 0)
        hcm_color = "#27ae60" if hcm_score >= 2 else ("#f39c12" if hcm_score >= 1 else "#e74c3c")

        st.markdown(f"""
        <div class="result-card" style="border-left:4px solid {hcm_color};">
            <h4 style="margin-top:0;color:#1a3a5c;">
                {ticker} — {meta_t['name']}
                <span class="sector-badge" style="float:right;">HCM Score: {hcm_score}/3</span>
            </h4>
        </div>
        """, unsafe_allow_html=True)

        col_mission, col_hcm = st.columns(2)

        with col_mission:
            st.markdown("**🎯 Mission Statement**")
            st.info(meta_t.get("mission", "Not specified"))

            st.markdown("**📋 Strategic Objectives**")
            objectives = ov_t.get("objectives", [])
            if objectives:
                for obj in objectives:
                    st.markdown(f"  - {clean_kg_node_name(obj)}")
            else:
                st.caption("No objectives found.")

        with col_hcm:
            st.markdown("**👥 HCM Metrics & Initiatives**")
            hcm_metrics = ov_t.get("hcm_metrics", [])
            initiatives = ov_t.get("initiatives", [])

            # Filter HCM-related initiatives
            hcm_initiatives = [i for i in initiatives
                              if any(kw in i.lower() for kw in
                                     ["talent", "workforce", "employee", "diversity",
                                      "inclusion", "belonging", "training", "human",
                                      "compensation", "benefit", "safety", "engagement",
                                      "retention", "apprentice", "intern", "culture",
                                      "health", "wellness"])]

            if hcm_metrics:
                st.markdown("*KG HCM Metrics:*")
                for m in hcm_metrics:
                    st.markdown(f"  - 📊 {clean_kg_node_name(m)}")

            if hcm_initiatives:
                st.markdown("*HCM-Related Initiatives:*")
                for init in hcm_initiatives:
                    st.markdown(f"  - 🚀 {clean_kg_node_name(init)}")
            elif not hcm_metrics:
                st.caption("No HCM disclosures found in KG.")

            # Show graph neighbors related to HCM
            G = load_kg_graph()
            hcm_kg_items = []
            for node in G.nodes():
                node_str = str(node)
                if ticker in node_str and G.nodes[node].get(":LABEL", "") in ("HumanCapitalMetric",):
                    text = G.nodes[node].get("text", "") or G.nodes[node].get("name", "") or node_str
                    hcm_kg_items.append(clean_kg_node_name(text))

            if hcm_kg_items:
                st.markdown("*Additional KG HCM Data:*")
                for item in hcm_kg_items[:5]:
                    st.markdown(f"  - 📈 {item}")

        st.markdown("---")

    # HCM Comparison Summary
    st.markdown("#### 📊 HCM Disclosure Depth — Cross-Company Comparison")
    hcm_data = []
    for ticker in sorted(COMPANIES.keys()):
        ov_t = get_company_overview_cached(ticker)
        scores_t = _compute_alignment_scores(ticker, ov_t)
        hcm_data.append({
            "Company": f"{ticker} — {COMPANIES[ticker]['name']}",
            "HCM Score": scores_t.get("HCM Disclosure Depth", 0),
            "HCM Metrics": len(ov_t.get("hcm_metrics", [])),
            "Total Initiatives": len(ov_t.get("initiatives", [])),
            "KG Nodes": ov_t["kg_stats"]["nodes"],
        })
    df_hcm = pd.DataFrame(hcm_data)
    st.dataframe(df_hcm, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 6: KG View — Enhanced Interactive Viewer
# ═══════════════════════════════════════════════════════════════════════
with tab6:
    st.info(
        "📖 **About this tab:** Interactive visualization of the company's **Knowledge Graph (KG)**. "
        "The KG is constructed from two sources: `kg1.ttl` (shared ontology with high-level Mission, "
        "Strategy, and Capability nodes) and the company-specific TTL file (detailed Risk Themes, "
        "Financial Metrics, and Initiatives extracted from the 10-K). Schema defined in `schema_1.owl`. "
        "Node types are color-coded — hover over nodes for details, hover edges for full RDF triplets. "
        "Use the navigation buttons and scroll to zoom. The graph shows a 2-hop subgraph from the company node."
    )

    try:
        import kg_viewer
        kg_viewer.render_kg_viewer(company, load_kg_graph, clean_display_text,
                                   format_risk_name, format_capability)
    except ImportError:
        # Fallback KG view
        st.markdown(f"### 🕸️ Knowledge Graph — {company}")
        G = load_kg_graph()

        # Show KG stats
        st.metric("Total Nodes", G.number_of_nodes())
        st.metric("Total Edges", G.number_of_edges())

        # Show node types
        label_counts = {}
        for n in G.nodes():
            l = G.nodes[n].get(":LABEL", "unlabeled")
            label_counts[l] = label_counts.get(l, 0) + 1
        st.markdown("#### Node Types")
        df_labels = pd.DataFrame(
            [{"Type": k, "Count": v} for k, v in
             sorted(label_counts.items(), key=lambda x: -x[1])[:15]]
        )
        st.dataframe(df_labels, use_container_width=True, hide_index=True)

        # Show neighbors of selected company
        rag = load_rag_engine()
        if rag:
            nodes = rag._find_company_nodes(G, company)
            if nodes:
                cn = nodes[0]
                st.markdown(f"#### Neighbors of `{cn}`")
                neighbor_data = []
                for n in G.neighbors(cn):
                    neighbor_data.append({
                        "Node": str(n),
                        "Type": G.nodes[n].get(":LABEL", "?"),
                        "Relation": G[cn][n].get("relation", "?"),
                    })
                if neighbor_data:
                    st.dataframe(pd.DataFrame(neighbor_data),
                                use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.8rem;">'
    'MSG-KG v2.0 | KG-RAG Portfolio Intelligence Interface | '
    'SEC 10-K Item 1 Analysis | MiniLM-L6-v2 + Knowledge Graph</div>',
    unsafe_allow_html=True,
)
