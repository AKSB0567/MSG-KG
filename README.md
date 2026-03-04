# MSG-KG: Portfolio Intelligence System 🧠📊

**[🚀 MSG-KG Live Demo](https://msg-kg.streamlit.app/)**

**Mission-Strategy-Goals Knowledge Graph (MSG-KG)** is a financial intelligence platform that combines **Retrieval-Augmented Generation (RAG)** with structured **Knowledge Graphs (KG)** to analyze corporate strategy alignment from SEC 10-K filings.

Built for **Finance Research**, this system ingests SEC 10-K filings, constructs an RDF-based knowledge graph of company missions, strategies, risks, and financials, and provides an interactive dashboard for comparative analysis.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Companies Analyzed](#-companies-analyzed)
- [Architecture](#-architecture)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Deployment](#-live-deployment-streamlit-cloud)

---

## 🚀 Key Features

### 1. 🏗️ Hybrid Graph-RAG Engine
- **Dual-Path Retrieval**: Combines vector search (`all-MiniLM-L6-v2` + FAISS) with structured KG subgraph extraction (NetworkX).
- **Cross-Encoder Reranking**: Re-scores combined results for high precision.
- **Context Fusion**: Merges textual evidence with KG structure into coherent context for LLM reasoning.
- **Explainable AI**: Returns **evidence spans** (text chunks with source), **KG path traces** (reasoning chains), and confidence scores.

### 2. 🕸️ RDF Knowledge Graph (TTL-Based)
- **rdflib TTL Parsing**: Full RDF Turtle file support with proper prefix resolution, `rdf:type` handling, and literal/object classification.
- **Dual KG Sources per Company**:
  - `kg1.ttl` — Shared ontology instances (Mission, Strategy, Capability nodes)
  - `{TICKER}.ttl` — Company-specific KG (Risk Themes, Financial Metrics, Initiatives)
- **Ontology**: Schema defined in `schema_1.owl` following the MSG-KG ontology pattern.
- **Interactive Visualization**: PyVis-powered graph with zoom, pan, edge triplet tooltips, fullscreen mode, and color-coded node legend.

### 3. ⚖️ Score-Based Comparative Analytics
- **8-Dimension Alignment Scorecard**: Automatically derived from KG node counts (no manual scoring):
  - Mission Clarity · Vision→Strategy Linkage · Strategy→Operations Grounding
  - Operations→Financial Linkage · HCM Disclosure Depth · Risk Awareness
  - Initiative Specificity · Capability Articulation
- **Portfolio Heatmap**: Plotly interactive heatmap (Viridis scale, 0–3 scoring).
- **Head-to-Head Comparison**: Side-by-side company alignment with winner indicators.

### 4. 👥 Mission → HCM Analysis
- **Human Capital Management**: Analyzes the connection between company mission and workforce practices.
- **HCM Scoring**: Derived from KG nodes related to workforce, diversity, training, compensation, and safety.
- **Cross-Company Comparison**: Tabular HCM disclosure depth comparison.

### 5. 🎨 Professional Enterprise UI
- **6-Tab Dashboard**: Overview · Ask KG-RAG · Competitive Landscape · Evidence Explorer · Mission→HCM Analysis · KG View
- **Sector-Based Filtering**: Filter companies by sector (replaces subjective Good/Bad classification).
- **Informational Context**: Every tab includes a "📖 About this tab" block explaining methodology and data sources.
- **Enterprise Theming**: Deep Red/Gold branding with custom CSS.

---

## 🏢 Companies Analyzed

| Ticker | Company | Sector | CIK | 10-K Filing |
|--------|---------|--------|-----|-------------|
| **AMD** | Advanced Micro Devices, Inc. | Semiconductors | `0000002488` | `0000002488-25-000012` |
| **ALX** | Alexander & Baldwin, Inc. | Real Estate / Diversified | `0000003499` | `0000003499-25-000004` |
| **LNG** | Cheniere Energy, Inc. | Energy / LNG | `0000003570` | `0000003570-25-000033` |

### KG Statistics

| Company | KG Nodes | KG Edges | Risk Themes | Financial Metrics | Initiatives |
|---------|----------|----------|-------------|-------------------|-------------|
| AMD | 181 | 179 | 94 | 60 | 7 |
| ALX | 140 | 138 | 54 | 50 | 4 |
| LNG | 143 | 141 | 24 | 77 | 4 |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MSG-KG Architecture v2.0                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SEC 10-K Filing ─┬─► sec_extractor.py ─► cleaned_10-K_*.txt    │
│                   │                                              │
│                   ├─► chunk_10k.py ─────► *_chunks.json          │
│                   │   (section-aware semantic chunking)           │
│                   │                                              │
│                   └─► kg_builder.py ────► *.ttl (KG)             │
│                       (LLM entity extraction + ontology)         │
│                                                                  │
├──────────────── RAG Pipeline (rag_engine.py) ───────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌────────────────┐   ┌──────────────────┐   │
│  │ Vector Path  │   │  Graph Path    │   │   Fusion Layer   │   │
│  │              │   │                │   │                  │   │
│  │ MiniLM-L6-v2 │   │ rdflib TTL     │   │ Cross-Encoder    │   │
│  │ FAISS Index  │   │ Entity Linking │   │ Reranker         │   │
│  │ Chunk Search │   │ Subgraph Ext.  │   │ Context Merge    │   │
│  └──────┬───────┘   └───────┬────────┘   └────────┬─────────┘   │
│         └───────────────────┼──────────────────────┘             │
│                             ▼                                    │
│                    Mixtral-8x7B (HF API)                         │
│                    Answer + Evidence + KG Paths                  │
│                                                                  │
├──────────────── Streamlit App (app.py) ──────────────────────────┤
│                                                                  │
│  Tab 1: Overview          │ Tab 4: Evidence Explorer             │
│  Tab 2: Ask KG-RAG        │ Tab 5: Mission → HCM Analysis       │
│  Tab 3: Competitive View  │ Tab 6: KG View (PyVis)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- [HuggingFace Account](https://huggingface.co/) with API token (for Mixtral-8x7B inference)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/AKSB0567/MSG-KG.git
   cd MSG-KG
   git checkout v2-kg-rag-pipeline
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate    # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```ini
   HUGGINGFACE_TOKEN=your_hf_token_here
   ```

---

## 🚦 Usage

### 1. Run the Data Pipeline (first time only)
```bash
# Generate semantic chunks from cleaned 10-K filings
python chunk_10k.py
```

### 2. Launch the Dashboard
```bash
streamlit run app.py
```
Access at `http://localhost:8501`.

### 3. Navigating the Interface

#### 🔍 Left Filter Panel
- **Sector Filter**: Filter companies by industry sector (Semiconductors, Real Estate, Energy).
- **Company Selector**: Choose the target company for analysis.
- **Data Sources**: Toggle between 10-K, Earnings, and News sources.
- **RAG Model Config**: Adjust chunk count, graph traversal depth, reranking, and explain mode.

#### 📊 Dashboard Tabs (Right Panel)

| Tab | Description |
|-----|-------------|
| **📊 Overview** | Company profile from KG: mission, objectives, capabilities, risk themes, financial metrics, HCM data. Includes CIK, filing ID, data source info, and SEC EDGAR link. |
| **🤖 Ask KG-RAG** | Natural-language Q&A with real-time pipeline progress. Returns model answer + evidence spans + KG path traces. |
| **⚖️ Competitive Landscape** | 8-dimension alignment heatmap, total alignment scores, head-to-head comparison with dimension-by-dimension scorecard. |
| **🔍 Evidence Explorer** | Browse raw 10-K text chunks with keyword and section filters. Shows chunk metadata (section, word range). |
| **👥 Mission → HCM** | Analyzes mission-to-HCM linkage per company. Shows HCM metrics, workforce initiatives, and cross-company HCM comparison. |
| **🕸️ KG View** | Interactive PyVis knowledge graph. Color-coded nodes, edge triplet tooltips, legend, fullscreen mode, and full triplet table. |

---

## 📂 Project Structure

```text
MSG-KG/
├── app.py                  # Streamlit application (6 tabs, enterprise UI)
├── rag_engine.py           # Hybrid Graph-RAG engine (Vector + KG retrieval)
├── kg_builder.py           # Knowledge Graph construction (LLM extraction)
├── kg_viewer.py            # Interactive KG visualization (PyVis)
├── chunk_10k.py            # Section-aware semantic chunking pipeline
├── sec_extractor.py        # SEC 10-K EDGAR crawling and parsing
├── __init__.py             # Package marker
├── requirements.txt        # Python dependencies (18 packages)
├── .env                    # API keys (not committed)
├── .env.example            # Environment template
├── MSGKG/                  # Knowledge Graph data
│   ├── kg1.ttl             # Shared ontology instances (all companies)
│   ├── AMD.ttl             # AMD company-specific KG
│   ├── ALX.ttl             # ALX company-specific KG
│   ├── LNG.ttl             # LNG company-specific KG
│   ├── schema_1.owl        # Ontology schema definition
│   ├── cleaned_10-K_*.txt  # Cleaned raw 10-K filings
│   ├── nodes.csv           # Legacy graph nodes (CSV fallback)
│   └── edges.csv           # Legacy graph edges (CSV fallback)
├── data/
│   ├── item1/              # Extracted Item 1 text per company
│   │   ├── AMD_item1.txt
│   │   ├── ALX_item1.txt
│   │   └── LNG_item1.txt
│   └── chunks/             # Semantic chunks for vector indexing
│       ├── AMD_chunks.json  # 20 chunks, ~9,768 words
│       ├── ALX_chunks.json  # 4 chunks, ~1,639 words
│       └── LNG_chunks.json  # 31 chunks, ~14,693 words
└── assets/
    └── custom.css          # Enterprise styling
```

---

## 🧠 Methodology

### Data Pipeline
1. **SEC Extraction**: `sec_extractor.py` pulls raw 10-K filings from EDGAR and cleans HTML artifacts.
2. **Semantic Chunking**: `chunk_10k.py` performs section-aware, sentence-level recursive chunking (~500 words/chunk). Sections: Business, Products, Competition, Human Capital, etc.
3. **Vectorization**: Chunks embedded via `all-MiniLM-L6-v2` (384-dim) and indexed in FAISS.
4. **KG Construction**: `kg_builder.py` extracts entities/relations using Mixtral-8x7B and encodes them as RDF Turtle files following the MSG-KG ontology.

### Knowledge Graph Structure
- **Ontology** (`schema_1.owl`): Defines classes — `Company`, `Mission`, `StrategicObjective`, `Capability`, `Initiative`, `RiskTheme`, `FinancialMetric`, `HumanCapitalMetric`.
- **Shared KG** (`kg1.ttl`): High-level company nodes with mission, strategy, and capability relationships.
- **Company KGs** (`AMD.ttl`, `ALX.ttl`, `LNG.ttl`): Detailed per-company data including all risk themes, financial metrics, and initiatives.
- **Parsing**: Uses `rdflib` for full Turtle syntax support (prefixes, `rdf:type`, literals, datatypes).

### Query Pipeline (Hybrid Graph-RAG)
When a user asks a question:
1. **Vector Search**: Retrieves top-k relevant text chunks from FAISS using MiniLM-L6-v2 embeddings.
2. **Graph Search**: Entity linking maps query terms to KG nodes; subgraph extraction traverses 2-hop neighborhoods.
3. **Reranking**: Cross-encoder reranker scores combined results by relevance.
4. **Context Fusion**: Merges text evidence + KG triplets into a coherent context with word budget control.
5. **Generation**: Mixtral-8x7B (HuggingFace Inference API) generates a grounded answer citing both text evidence and KG paths.

### Alignment Scoring
Each company is scored on 8 dimensions (0–3 scale) **automatically derived from KG data**:

| Dimension | Scoring Logic |
|-----------|---------------|
| Mission Clarity | Mission node length > 20 chars → 3 |
| Vision → Strategy Linkage | min(3, count of StrategicObjective nodes) |
| Strategy → Operations Grounding | min(3, count of Capability + Initiative nodes) |
| Operations → Financial Linkage | FinancialMetric count: ≥50 → 3, ≥10 → 2, >0 → 1 |
| HCM Disclosure Depth | HumanCapitalMetric count: ≥3 → 3, ≥1 → 2 |
| Risk Awareness & Mitigation | RiskTheme count: ≥10 → 3, ≥3 → 2, >0 → 1 |
| Initiative Specificity | Initiative count: ≥5 → 3, ≥2 → 2, >0 → 1 |
| Capability Articulation | Capability count: ≥3 → 3, ≥1 → 2 |

---

## 🌐 Live Deployment (Streamlit Cloud)

### Deploy to Streamlit Community Cloud
1. **Authorize**: Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
2. **Deploy**: Select repository `AKSB0567/MSG-KG`, branch `v2-kg-rag-pipeline`, and `app.py` as the main file.
3. **Secrets**: In the Streamlit Cloud dashboard, go to **Settings > Secrets** and add:
   ```toml
   HUGGINGFACE_TOKEN = "your_token_here"
   ```

### Requirements for Cloud Deployment
- All data files (TTLs, chunks, item1 texts) are committed to the repository.
- The app loads KG from TTL files at startup (no external database required).
- HuggingFace token is required only for the "Ask KG-RAG" tab (LLM generation).

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.30+ |
| **Embeddings** | `all-MiniLM-L6-v2` (sentence-transformers) |
| **Vector Index** | FAISS (faiss-cpu) |
| **KG Storage** | RDF Turtle (.ttl) via rdflib |
| **Graph Engine** | NetworkX (DiGraph) |
| **Reranker** | Cross-Encoder (sentence-transformers) |
| **LLM** | Mixtral-8x7B (HuggingFace Inference API) |
| **Visualization** | Plotly (heatmaps) · PyVis (KG graph) |
| **KG Extraction** | Mixtral-8x7B + Rule-based fallback |

---

## 📜 License
Copyright (c) 2026 MSG-KG Contributors. All Rights Reserved.
Privileged & Confidential — Enterprise Internal Use Only.
