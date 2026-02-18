"""
MSG-KG v1.0 — KG-RAG Portfolio Intelligence Interface
Streamlit application with 5 tabs: Overview, Ask KG-RAG, Comparison,
Evidence Explorer, KG View.
"""

import streamlit as st
"""
MSG-KG: Portfolio Intelligence Dashboard
========================================
Main Streamlit application that provides a professional interface for 
exploring the Mission-Strategy-Goals Knowledge Graph.

Features:
- Full-width corporate header & footer
- Amazon-style left-sidebar filter panel
- Hybrid RAG (Vector + Graph) exploration
- Comparative strategy scorecards & heatmaps
"""

import os, re, json, pathlib, sys
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSG-KG v1.0 — KG-RAG Portfolio Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
ITEM1_DIR  = BASE_DIR / "data" / "item1"
KG_DIR     = BASE_DIR / "MSGKG"

# ── Custom CSS Loader ────────────────────────────────────────────────────
def load_css(file_name):
    """
    Injects custom CSS from a local file into the Streamlit app.
    Used for overriding default styles and implementing the corporate theme.
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = BASE_DIR / "assets" / "custom.css"
if css_path.exists():
    load_css(str(css_path))

COMPANIES = {
    "AAPL": {
        "name": "Apple Inc.",
        "fiscal_year": "FY2024",
        "category": "Bad",
        "mission": "To bring the best user experience to customers through innovative hardware, software, and services.",
        "alignment_label": "Worst-Alignment",
        "alignment_rationale": "Mission and purpose language is sparse or diffuse in regulated filings, limiting traceable connections between stated intent, operational actions, and financial indicators.",
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "fiscal_year": "FY2024",
        "category": "Bad",
        "mission": "To build great products that accelerate next-generation computing experiences through high-performance and adaptive computing technology.",
        "alignment_label": "Worst-Alignment",
        "alignment_rationale": "High-level innovation and leadership claims appear weakly grounded in explicit strategic objectives or financial pathways in the sampled disclosures.",
    },
    "TGT": {
        "name": "Target Corporation",
        "fiscal_year": "FY2025",
        "category": "Bad",
        "mission": "To help all families discover the joy of everyday life by delivering an experience that is uniquely Target.",
        "alignment_label": "Worst-Alignment",
        "alignment_rationale": "Broad purpose statements emphasizing community and guest experience exhibit limited operational and financial linkage in disclosed strategy and performance metrics.",
    },
    "WMT": {
        "name": "Walmart Inc.",
        "fiscal_year": "FY2025",
        "category": "Good",
        "mission": "To help people save money and live better — through everyday low prices, powered by everyday low cost.",
        "alignment_label": "Best-Alignment",
        "alignment_rationale": "Purpose language centered on affordability and access is operationalized through omni-channel strategy, logistics modernization, and pricing disclosures tied to customer volume and margin performance.",
    },
    "TSN": {
        "name": "Tyson Foods Inc.",
        "fiscal_year": "FY2025",
        "category": "Good",
        "mission": "To raise the world's expectations for how much good food can do — feeding people sustainably, responsibly, and well.",
        "alignment_label": "Best-Alignment",
        "alignment_rationale": "Mission statements on food access and safety align with disclosures on supply chain controls, product portfolio strategy, and risk mitigation tied to commodity and regulatory factors.",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "fiscal_year": "FY2025",
        "category": "Good",
        "mission": "To empower every person and every organization on the planet to achieve more.",
        "alignment_label": "Best-Alignment",
        "alignment_rationale": "Vision statements emphasizing empowerment and cloud-first strategy align with sustained disclosures on infrastructure investment, AI deployment, and recurring revenue growth in cloud and enterprise services.",
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

ALIGNMENT_SCORES = {
    "MSFT": {"Mission Clarity": 3, "Vision → Strategy Linkage": 3, "Strategy → Operations Grounding": 3,
             "Operations → Financial Linkage": 3, "HCM Disclosure Depth": 2, "Risk Awareness & Mitigation": 3,
             "Initiative Specificity": 3, "Capability Articulation": 3},
    "WMT":  {"Mission Clarity": 3, "Vision → Strategy Linkage": 3, "Strategy → Operations Grounding": 3,
             "Operations → Financial Linkage": 3, "HCM Disclosure Depth": 2, "Risk Awareness & Mitigation": 3,
             "Initiative Specificity": 2, "Capability Articulation": 3},
    "TSN":  {"Mission Clarity": 3, "Vision → Strategy Linkage": 2, "Strategy → Operations Grounding": 3,
             "Operations → Financial Linkage": 2, "HCM Disclosure Depth": 2, "Risk Awareness & Mitigation": 3,
             "Initiative Specificity": 2, "Capability Articulation": 2},
    "TGT":  {"Mission Clarity": 2, "Vision → Strategy Linkage": 1, "Strategy → Operations Grounding": 1,
             "Operations → Financial Linkage": 1, "HCM Disclosure Depth": 1, "Risk Awareness & Mitigation": 2,
             "Initiative Specificity": 1, "Capability Articulation": 1},
    "AMD":  {"Mission Clarity": 2, "Vision → Strategy Linkage": 1, "Strategy → Operations Grounding": 2,
             "Operations → Financial Linkage": 1, "HCM Disclosure Depth": 1, "Risk Awareness & Mitigation": 2,
             "Initiative Specificity": 2, "Capability Articulation": 2},
    "AAPL": {"Mission Clarity": 1, "Vision → Strategy Linkage": 1, "Strategy → Operations Grounding": 2,
             "Operations → Financial Linkage": 1, "HCM Disclosure Depth": 1, "Risk Awareness & Mitigation": 2,
             "Initiative Specificity": 2, "Capability Articulation": 1},
}


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
        box_shadow: 0 6px 12px rgba(0,0,0,0.08);
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
        letter-spacing: 0.05rem;
        font-weight: 600;
    }

    /* Panel Header */
    .panel-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2c3e50;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .sidebar-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }

    /* Overview cards */
    .overview-card {
        background: linear-gradient(135deg, #f8f9fa, #fff);
        border: 1px solid #e0e5ec;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .overview-card h4 {
        color: #2962a8;
        margin-top: 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
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


# ── Header Helper ────────────────────────────────────────────────────────
# ── UI COMPONENT RENDERING ──────────────────────────────────────────────────

def render_header():
    """
    Renders the consistent global dashboard header with corporate branding.
    Uses a solid background and flexbox for alignment.
    """
    # Professional Hero Header replacing older gradient style
    st.markdown("""
    <div style="background-color:#CD1515; padding:15px 20px; border-bottom:2px solid #AA8F00; margin-bottom:20px; display:flex; align-items:center; justify-content:space-between; box-shadow:0 2px 5px rgba(0,0,0,0.15);">
        <div style="display:flex; align-items:center;">
             <!-- Logo Area -->
             <div style="font-size:26px; font-weight:700; color:#FFFFFF; margin-right:15px; letter-spacing:-0.5px;">
                 MSG-KG
             </div>
             <div style="border-left:1px solid #FFFFFF50; padding-left:15px; font-size:16px; font-weight:500; color:#FFFFFFee;">
                 Portfolio Intelligence System
             </div>
        </div>
        <!-- Right Utility Area -->
        <div style="text-align:right;">
             <div style="font-size:12px; font-weight:600; color:#FFFFFFcc; letter-spacing:0.5px; text-transform:uppercase;">
                 Enterprise Edition v1.2
             </div>
             <div style="font-size:11px; color:#FFFFFFaa; margin-top:2px;">
                 Secure Connection <span style="color:#FFF;">●</span>
             </div>
        </div>
    </div>
    """, unsafe_allow_html=True)





# ── Display text cleanup utilities ───────────────────────────────────────

# Map truncated KG names to proper display names
RISK_NAME_MAP = {
    "competi": "Competition",
    "regulat": "Regulatory Compliance",
    "economic": "Economic & Macro Conditions",
    "government": "Government & Policy",
    "inventory": "Inventory & Supply Chain",
    "cybersecurity": "Cybersecurity",
    "workforce": "Workforce & Talent",
    "compliance": "Regulatory Compliance",
    "supply chain": "Supply Chain Disruption",
    "supply-chain": "Supply Chain Disruption",
    "inflation": "Inflation & Cost Pressures",
    "legal": "Legal & Litigation",
}


def clean_display_text(text: str) -> str:
    """
    Clean KG-extracted text for display:
    - Remove leading bullets/markers
    - Ensure sentence ends properly
    - Collapse multi-line artifacts
    - Strip form/page references
    """
    if not text or not text.strip():
        return ""

    # Collapse newlines into spaces
    t = " ".join(text.split())

    # Remove leading bullet chars
    t = t.lstrip("•·-– ")

    # Remove leading page/form refs like "| 2024 Form 10-K | 2"
    import re as _re
    t = _re.sub(r'^\|?\s*\d{4}\s+Form\s+10-K\s*\|?\s*\d*\s*', '', t).strip()

    # Capitalize first letter
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    # Ensure the sentence ends properly (add period if cut off)
    if t and t[-1] not in '.!?:"\'':
        t = t.rstrip(',;') + '.'

    return t


def format_risk_name(raw_name: str) -> str:
    """Map truncated risk theme names to proper display names."""
    key = raw_name.strip().lower()
    return RISK_NAME_MAP.get(key, raw_name.strip().title())


def format_capability(name: str) -> str:
    """Capitalize capability names for display."""
    return name.strip().title() if name else ""


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


def load_kg_graph() -> nx.DiGraph:
    """Load KG from CSVs."""
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

# ── Common Header ────────────────────────────────────────────────────────
render_header()

# ── MAIN LAYOUT CONSTRUCTION ────────────────────────────────────────────────

# We use a 2-column layout to mimic the "Amazon" interface:
# Left Column: Global controls and filters
# Right Column: Main content display and tabs
col_filters, col_content = st.columns([1, 4])

# ── Left Column: Contextual Filters (Amazon Style) ───────────────────────
with col_filters:
    st.markdown("### 🔍 Filters")
    st.markdown("---")
    
    # Portfolio
    st.caption("PORTFOLIO CONTEXT")
    portfolio = st.radio("Category", ["All", "Good", "Bad"], 
                         horizontal=False, label_visibility="collapsed")
    
    if portfolio == "All":
        available_tickers = list(COMPANIES.keys())
    else:
        available_tickers = [t for t, m in COMPANIES.items() if m["category"] == portfolio]
        
    # Company
    st.caption("SELECT COMPANY")
    company = st.selectbox("Company", available_tickers, 
                           format_func=lambda t: f"{t}",
                           label_visibility="collapsed")
                           
    fiscal_year = COMPANIES[company]["fiscal_year"]
    st.info(f"Fiscal Year: **{fiscal_year}**")
    
    st.markdown("---")
    
    # Evidence Sources
    st.caption("DATA SOURCES")
    evidence_sources = st.multiselect("Sources", ["10-K", "Earnings", "News"], 
                                      default=["10-K"], label_visibility="collapsed")
                                      
    st.markdown("---")
    
    # Model Params (Collapsible to save space if needed)
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
    # Tabs within the right content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🤖 Ask KG-RAG",
        "⚖️ Competitive Landscape",
        "🔍 Evidence Explorer",
        "🕸️ KG View",
    ])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"### Company Overview: **{company}** — {COMPANIES[company]['name']}")

    G = load_kg_graph()
    company_node = f"Company:{company}"

    # Gather KG data
    mission = ""
    objectives = []
    capabilities = []
    initiatives = []
    risks = []
    node_count = 0
    edge_count = 0

    if company_node in G:
        for neighbor in G.neighbors(company_node):
            label = G.nodes[neighbor].get(":LABEL", "")
            text = G.nodes[neighbor].get("text", "") or G.nodes[neighbor].get("name", "")
            node_count += 1
            edge_count += 1

            if label == "Mission":
                mission = text
            elif label == "StrategicObjective":
                objectives.append(text)
            elif label == "Capability":
                capabilities.append(text)
            elif label == "Initiative":
                initiatives.append(text)
            elif label == "RiskTheme":
                risks.append(text)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🏢 Ticker", company)
    with col2:
        st.metric("📅 Fiscal Year", COMPANIES[company]["fiscal_year"])
    with col3:
        st.metric("🔗 KG Nodes", node_count + 1)
    with col4:
        st.metric("➡️ KG Edges", edge_count)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    # Use curated mission as primary, fall back to KG-extracted
    curated_mission = COMPANIES[company].get("mission", "")
    display_mission = curated_mission if curated_mission else mission
    align_label = COMPANIES[company].get("alignment_label", "")
    align_rationale = COMPANIES[company].get("alignment_rationale", "")

    with col_l:
        # ── Mission Card (enriched, bigger) ──
        label_color = "#27ae60" if "Best" in align_label else "#e74c3c"
        st.markdown(f"""
        <div class="result-card" style="border-top:4px solid {label_color};">
            <h4 style="margin-top:0;color:#1a3a5c;">🎯 Mission Statement</h4>
            <div style="background:linear-gradient(135deg,#f0f4f8,#e8eef5);padding:1rem 1.2rem;
                        border-radius:8px;margin:0.5rem 0;font-size:1.05rem;line-height:1.6;
                        border-left:4px solid {label_color};">
                <em>"{display_mission}"</em>
            </div>
            <div style="margin-top:0.6rem;">
                <span style="background:{label_color};color:white;padding:4px 14px;
                             border-radius:14px;font-size:0.82rem;font-weight:600;">
                    {align_label}</span>
            </div>
            <p style="color:#666;font-size:0.82rem;margin-top:0.5rem;line-height:1.5;">
                {align_rationale}
            </p>
        </div>""", unsafe_allow_html=True)

        # ── Strategic Objectives (cleaned) ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Strategic Objectives")
        if objectives:
            for obj in objectives:
                cleaned = clean_display_text(obj)
                if cleaned:
                    st.markdown(f"- {cleaned}")
        else:
            st.caption("No objectives extracted yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        # ── Capabilities (formatted) ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Capabilities")
        if capabilities:
            for cap in capabilities:
                st.markdown(f"- {format_capability(cap)}")
        else:
            st.caption("No capabilities extracted yet.")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Risk Themes (fixed names) ──
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### ⚠️ Risk Themes")
        if risks:
            for risk in risks:
                st.markdown(f"- {format_risk_name(risk)}")
        else:
            st.caption("No risks extracted yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Initiatives (cleaned, full-width) ──
    if initiatives:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 🚀 Initiatives")
        for init in initiatives:
            cleaned = clean_display_text(init)
            if cleaned:
                st.markdown(f"- {cleaned}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Filing metadata
    st.markdown("---")
    st.markdown("#### 📄 Filing Metadata")
    item1_path = ITEM1_DIR / f"{company}_item1.txt"
    if item1_path.exists():
        text = item1_path.read_text(encoding="utf-8")
        st.success(f"Item 1 text loaded: **{len(text):,}** characters")
    else:
        st.warning("Item 1 text not found. Run `python sec_extractor.py` first.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: Ask KG-RAG
# ═══════════════════════════════════════════════════════════════════════
with tab2:
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
                ("🔎 Vector Retrieval (FinBERT embedding + FAISS search)...", 0.15),
                ("🕸️ Graph Retrieval (NetworkX KG traversal)...", 0.30),
                ("📊 Reranking (Cross-encoder scoring)...", 0.50),
                ("🤖 KG-RAG Reasoning (Mixtral-8x7B generation)...", 0.75),
            ]

            # Show progress bar with stages
            progress_bar = progress_placeholder.progress(0, text="Initializing pipeline...")
            for stage_text, stage_pct in pipeline_stages:
                elapsed = _time.time() - t_start
                progress_bar.progress(stage_pct, text=f"{stage_text}  ⏱️ {elapsed:.1f}s")
                if stage_pct < 0.50:
                    _time.sleep(0.3)  # Brief visual pause for fast stages

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
                    # Clean evidence text for display
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
            st.error("RAG engine not loaded. Make sure all dependencies are installed and data files exist.")

    else:
        # Show placeholder
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
    # ── 1. Portfolio Heatmap Section (Prominent, Interactive) ──────
    st.markdown('<div class="panel-header">1. Portfolio Alignment Heatmap</div>', unsafe_allow_html=True)
    st.caption("Score intensity (0-3): **0**=Missing · **1**=Weak · **2**=Moderate · **3**=Strong")

    # Heatmap Controls
    h_col1, h_col2, h_col3, h_col4 = st.columns([1, 1.2, 1.5, 0.8])
    with h_col1:
        hm_cats = st.multiselect("Filter by Category:", ["All", "Good", "Bad"], default=["All"])
    with h_col2:
        hm_sort = st.selectbox("Sort Heatmap by:", ["Total Score (High → Low)", "Total Score (Low → High)", "Ticker (A-Z)"])
    with h_col3:
        # User requested ability to adding/removing categories (features on x-axis)
        hm_dims = st.multiselect("Select Dimensions:", ALIGNMENT_DIMENSIONS, default=ALIGNMENT_DIMENSIONS)
    with h_col4:
        # Dynamic resizing toggle
        st.write("") # Spacer
        fit_width = st.checkbox("Fit to Width", value=False, help="Uncheck to enable scrolling for better label visibility")
    
    if not hm_dims:
        st.warning("Please select at least one dimension.")
        st.stop()

    # Build Heatmap Data
    all_tickers = sorted(COMPANIES.keys())
    heat_rows = []
    for t in all_tickers:
        meta = COMPANIES[t]
        # logic: if All is selected, show everything. otherwise filter by specific category
        if "All" not in hm_cats and meta.get("category", "Good") not in hm_cats:
            continue
        
        row = {"Ticker": t, "Category": meta.get("category", "")}
        scores = ALIGNMENT_SCORES.get(t, {})
        row["Total"] = sum(scores.values())
        for dim in hm_dims:
            row[dim] = scores.get(dim, 0)
        heat_rows.append(row)
    
    if heat_rows:
        df_heat = pd.DataFrame(heat_rows)
        # Sort Logic
        if "High → Low" in hm_sort:
            df_heat = df_heat.sort_values("Total", ascending=False)
        elif "Low → High" in hm_sort:
            df_heat = df_heat.sort_values("Total", ascending=True)
        else:
            df_heat = df_heat.sort_values("Ticker", ascending=True)
        
        df_display = df_heat.set_index("Ticker").drop(columns=["Category", "Total"])
        
        # Render Interactive Heatmap using Plotly (Better zoom/pan/resize support)
        import plotly.graph_objects as go
        
        # Create hover text
        hover_text = []
        for index, row in df_display.iterrows():
            hover_text.append([f"Ticker: {index}<br>Dimension: {col}<br>Score: {val}" for col, val in row.items()])

        # Dynamic Dimensions
        cell_height = 40
        cell_width = 120 # min width for readable labels
        
        # Calculate figure dimensions
        n_rows = len(df_display)
        n_cols = len(df_display.columns)
        
        fig_height = max(400, n_rows * cell_height + 100) # +100 for margins
        fig_width = n_cols * cell_width + 150 # +150 for y-axis labels
        
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
            xgap=1, ygap=1
        ))

        fig.update_layout(
            title_text=None,
            height=fig_height,
            xaxis=dict(
                side="bottom",
                tickangle=-45,
                tickmode='linear', # force all ticks
            ),
            yaxis=dict(
                autorange="reversed", # top-to-bottom
                tickmode='linear', # force all ticks
            ),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        
        if fit_width:
            # Responsive: Fit container width (Plotly handles wrapping well, but labels might be tight)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            # Scrollable: Explicit width for readability
            fig.update_layout(width=fig_width)
            st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': True})
    else:
        st.info("No companies match the selected filter.")

    st.markdown("---")

    # ── 2. Detailed 2-Company Comparison (Restored Logic) ────────────
    st.markdown('<div class="panel-header">2. Head-to-Head Comparison</div>', unsafe_allow_html=True)
    st.caption("Select two companies to compare their specific Alignment Checklist and Evidence Chain.")

    available_tickers = sorted(COMPANIES.keys())
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        company1 = st.selectbox("Company 1", available_tickers, index=0, key="cmp1",
                                format_func=lambda t: f"{t} — {COMPANIES[t]['name']}")
    with col_c2:
        default_idx = min(3, len(available_tickers) - 1)  # default to WMT if available
        company2 = st.selectbox("Company 2", available_tickers, index=default_idx, key="cmp2",
                                format_func=lambda t: f"{t} — {COMPANIES[t]['name']}")

    # Always show comparison (no button required)
    meta1 = COMPANIES[company1]
    meta2 = COMPANIES[company2]
    scores1 = ALIGNMENT_SCORES.get(company1, {})
    scores2 = ALIGNMENT_SCORES.get(company2, {})

    # ── Header cards ─────────────────────────────────────────────────
    hdr1, hdr2 = st.columns(2)
    with hdr1:
        cat_color1 = "#27ae60" if meta1["category"] == "Good" else "#e74c3c"
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {cat_color1};">
            <h3 style="border:none;padding:0;">{company1} — {meta1['name']}</h3>
            <span style="background:{cat_color1};color:white;padding:2px 10px;border-radius:12px;
                         font-size:0.75rem;font-weight:600;">{meta1.get('category','')}</span>
            <p style="margin-top:0.6rem;font-size:0.88rem;color:#555;">
                📅 {meta1['fiscal_year']} &nbsp;|&nbsp;
                🎯 <em>{meta1.get('mission','')}</em>
            </p>
        </div>""", unsafe_allow_html=True)
    with hdr2:
        cat_color2 = "#27ae60" if meta2["category"] == "Good" else "#e74c3c"
        st.markdown(f"""<div class="result-card" style="border-top:4px solid {cat_color2};">
            <h3 style="border:none;padding:0;">{company2} — {meta2['name']}</h3>
            <span style="background:{cat_color2};color:white;padding:2px 10px;border-radius:12px;
                         font-size:0.75rem;font-weight:600;">{meta2.get('category','')}</span>
            <p style="margin-top:0.6rem;font-size:0.88rem;color:#555;">
                📅 {meta2['fiscal_year']} &nbsp;|&nbsp;
                🎯 <em>{meta2.get('mission','')}</em>
            </p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Overall Alignment Score ──────────────────────────────────────
    total1 = sum(scores1.values()) if scores1 else 0
    total2 = sum(scores2.values()) if scores2 else 0
    max_score = len(ALIGNMENT_DIMENSIONS) * 3
    pct1 = round(total1 / max_score * 100) if max_score else 0
    pct2 = round(total2 / max_score * 100) if max_score else 0

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

    # ── Dimension-by-dimension scorecard ─────────────────────────────
    st.markdown("#### 📋 Alignment Scorecard — Dimension Checklist")
    st.caption("Each dimension scored 0–3: 0 = Not present · 1 = Weak · 2 = Moderate · 3 = Strong")

    score_icons = {0: "⬜", 1: "🟡", 2: "🟠", 3: "🟢"}

    # Build comparison table
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

    # ── Detailed Evidence Logic (Deep Dive) ────────────────────────
    # Use existing helper logic
    G = load_kg_graph()

    def get_alignment_path(ticker):
        cn = f"Company:{ticker}"
        path = {"Mission": "", "Objectives": [], "Capabilities": [], "Risks": [],
                "Initiatives": []}
        # Use curated mission
        path["Mission"] = COMPANIES[ticker].get("mission", "")
        if cn in G:
            for nb in G.neighbors(cn):
                label = G.nodes[nb].get(":LABEL", "")
                text = G.nodes[nb].get("text", "") or G.nodes[nb].get("name", "")
                if label == "StrategicObjective":
                    path["Objectives"].append(clean_display_text(text[:120]))
                elif label == "Capability":
                    path["Capabilities"].append(format_capability(text))
                elif label == "RiskTheme":
                    path["Risks"].append(format_risk_name(text))
                elif label == "Initiative":
                    path["Initiatives"].append(clean_display_text(text[:100]))
        return path

    p1 = get_alignment_path(company1)
    p2 = get_alignment_path(company2)

    ep1, ep2 = st.columns(2)

    def render_evidence_path(col, ticker, path_data, meta):
        with col:
            cat_c = "#27ae60" if meta["category"] == "Good" else "#e74c3c"
            st.markdown(f"**{ticker}** — {meta['name']}")

            # Mission → Strategy → Operations → Financial chain
            steps = [
                ("🎯 Mission/Vision", path_data["Mission"]),
                ("📋 Strategic Goals", " · ".join(path_data["Objectives"][:3]) if path_data["Objectives"] else "—"),
                ("🛠️ Operational Capabilities", " · ".join(path_data["Capabilities"][:4]) if path_data["Capabilities"] else "—"),
                ("🚀 Initiatives", " · ".join(path_data["Initiatives"][:3]) if path_data["Initiatives"] else "—"),
                ("⚠️ Risk Mitigation", " · ".join(path_data["Risks"][:3]) if path_data["Risks"] else "—"),
            ]

            for label, val in steps:
                st.markdown(f"""
                <div class="evidence-item" style="border-left-color:{cat_c};">
                    <strong>{label}</strong><br>
                    <span style="color:#555;">{val}</span>
                </div>
                """, unsafe_allow_html=True)

    render_evidence_path(ep1, company1, p1, meta1)
    render_evidence_path(ep2, company2, p2, meta2)

# TAB 4: Evidence Explorer
# ═══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"### Evidence Explorer — {company}")

    df_chunks = load_chunks_df(company)

    if not df_chunks.empty:
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            keyword = st.text_input("🔎 Filter by keyword:", key="ev_keyword")
        with col_f2:
            page_max = max(2, int(df_chunks["page_estimate"].max()))
            page_filter = st.slider("📄 Page range:",
                                     min_value=1,
                                     max_value=page_max,
                                     value=(1, page_max),
                                     key="ev_pages")

        # Apply filters
        filtered = df_chunks.copy()
        if keyword:
            filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]
        filtered = filtered[
            (filtered["page_estimate"] >= page_filter[0]) &
            (filtered["page_estimate"] <= page_filter[1])
        ]

        st.markdown(f"**Showing {len(filtered)} of {len(df_chunks)} chunks**")

        for _, row in filtered.iterrows():
            with st.expander(f"📄 {row['chunk_id']} — Page ~{row['page_estimate']}"):
                st.markdown(row["text"])
                st.caption(f"Words {row['start_word']}–{row['end_word']} | "
                          f"Section: {row['section']}")
    else:
        st.warning(f"No chunks found for {company}. Run `python sec_extractor.py` first.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 5: KG View — Enhanced Interactive Viewer
# ═══════════════════════════════════════════════════════════════════════
with tab5:
    try:
        import kg_viewer
        kg_viewer.render_kg_viewer(company, load_kg_graph, clean_display_text,
                                   format_risk_name, format_capability)
    except ImportError:
        st.error("KG Viewer module missing.")


# ── Footer ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.8rem;">'
    'MSG-KG v1.0 | KG-RAG Portfolio Intelligence Interface | '
    'SEC 10-K Item 1 Analysis | FinBERT + Knowledge Graph</div>',
    unsafe_allow_html=True,
)
