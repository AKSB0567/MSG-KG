---
title: MSG-KG
emoji: 📊
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
---

# MSG-KG: Mission Statement Knowledge Graph Intelligence

**Mission-Strategy-Goals Knowledge Graph (MSG-KG)** is a financial intelligence platform that extracts, evaluates, and compares corporate mission statements from **91 S&P company SEC 10-K filings** using a hybrid **RAG + Knowledge Graph** pipeline with **FIBO ontology alignment**.

Built for **Finance Research**, this system provides evidence-based mission statement analysis with interactive visualizations, multi-schema knowledge graphs, and portfolio-level comparative analytics.

---

## Table of Contents

- [Key Features](#key-features)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Mission Extraction Pipeline](#mission-extraction-pipeline)
- [Knowledge Graph Schemas](#knowledge-graph-schemas)
- [Mission Evaluation Engine](#mission-evaluation-engine)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Key Features

### 1. Mission Statement Extraction Pipeline
- **91 companies** processed from SEC 10-K filings (87 with missions, 4 genuinely absent)
- **3-tier classification**: Explicit (Tier 1), Implied (Tier 2), Inferred (Tier 3)
- SEC SGML header stripping + Item 1 section extraction
- Sliding window keyword scoring with HCM section deprioritization
- Confidence scoring and evidence provenance tracking

### 2. Dual-Schema Mission Knowledge Graphs
- **General Schema (LLM-Driven)**: Flat Company -> Mission graph per company
- **Ontology Schema (FIBO-Aligned)**: Mission decomposed into Stakeholders, Values, Objectives, Capabilities, and Business Domains
- **182 TTL files** (91 general + 91 ontology) with 280 total triplets
- Interactive PyVis visualization with schema switching dropdown

### 3. 8-Dimension Mission Quality Evaluation
- **Clarity of Purpose** | **Stakeholder Focus** | **Value Proposition** | **Actionability**
- **Innovation Signal** | **Semantic Completeness** | **Internal Consistency** | **Writing Quality**
- 0-4 scale: Outstanding / Strong / Adequate / Weak / Missing
- Buzzword detection, stakeholder gap analysis, semantic component analysis

### 4. Evidence-Based Reasoning
- Highlighted evidence from actual 10-K text files matched by CIK
- Source file name, extraction tier, confidence score displayed
- Keyword-density matching to find mission passages in 3-23MB documents
- Raw source text viewer with contextual highlighting

### 5. Portfolio Comparison Analytics
- Interactive heatmap across all 91 companies and 8 dimensions
- Sector comparison (average scores per industry)
- Top/bottom performer rankings
- Head-to-head company comparison
- CSV export for external analysis

### 6. SIC Code Normalization
- 200+ SIC code mappings (e.g., `6022` -> `State Commercial Banks`)
- All sectors displayed as `Code - Industry Name` format
- Sector-based filtering in the dashboard

---

## End-to-End Pipeline

The system processes SEC 10-K filings through 6 stages:

### Stage 1: Data Ingestion
```
SEC EDGAR -> sec_extractor.py -> cleaned_10-K_*.txt (91 files, 3-23MB each)
```
- Raw 10-K filings downloaded from SEC EDGAR by CIK
- HTML artifacts cleaned, text extracted
- Files stored in `MSGKG/data/data/`

### Stage 2: Knowledge Graph Construction
```
cleaned_10-K_*.txt -> kg_builder.py -> *.ttl (91 company KGs + schema.ttl)
```
- LLM-based entity extraction (Mixtral-8x7B)
- RDF Turtle encoding following MSG-KG ontology
- Entity types: Company, Mission, Capability, Initiative, RiskTheme, FinancialMetric, HCM
- Output in `MSGKG/data/output/`

### Stage 3: Mission Statement Extraction
```
cleaned_10-K_*.txt + *.ttl -> mission_extractor.py -> companies_registry.json
```
- **Priority 1**: Use KG missions from TTL files (11 companies with `sec:hasMission`)
- **Priority 2**: Extract from 10-K text using 4-step process:
  1. Strip SEC SGML headers (`</SEC-HEADER>` detection)
  2. Extract Item 1 (Business) section via regex
  3. Sliding window scan (600 chars, 250 step) with tiered keyword scoring
  4. HCM section deprioritization to avoid HR false positives
- **Output**: Mission text, tier (1/2/3), confidence (0-1), evidence chunk, source

### Stage 4: Mission KG Generation
```
companies_registry.json -> mission_kg_builder.py -> data/mission_kg/{general,ontology}/*.ttl
```
- **General Schema**: Simple Company -> MissionStatement graph
- **Ontology Schema**: FIBO-aligned decomposition using rule-based NLP:
  - Stakeholders: keyword matching against 6 stakeholder groups
  - Values: 8 value categories (Innovation, Sustainability, Integrity, etc.)
  - Objectives: action-verb phrase extraction via regex
  - Domains: industry detection from mission + sector info
  - Capabilities: pulled from existing KG registry data

### Stage 5: Registry Building
```
*.ttl + cleaned_10-K_*.txt -> build_registry.py -> companies_registry.json
```
- Pre-computes company metadata, KG stats, alignment scores
- Mission fallback chain: KG -> Item 1 text -> cleaned 10-K text
- SIC code normalization via `sic_codes.py`
- Output: 91 companies with mission, sector, scores, evidence metadata

### Stage 6: Interactive Dashboard
```
companies_registry.json + *.ttl -> app.py (Streamlit) -> http://localhost:8501
```
- 4-tab dashboard: Mission Statement | Evidence-Based Reasoning | Portfolio Comparison | Knowledge Graph
- Real-time mission evaluation (8 dimensions, 0-4 scale)
- KG schema switching (Existing / LLM-Driven / Ontology-Driven)
- Evidence highlighting from 10-K source files

---

## Architecture

```
+---------------------------------------------------------------------+
|                    MSG-KG Architecture v3.0                           |
+---------------------------------------------------------------------+
|                                                                       |
|  SEC 10-K Filing --+--> sec_extractor.py ----> cleaned_10-K_*.txt    |
|                    |                                                  |
|                    +--> kg_builder.py --------> *.ttl (SEC KG)       |
|                    |                                                  |
|                    +--> mission_extractor.py -> companies_registry    |
|                    |    (keyword scoring,       .json                 |
|                    |     tier classification)                         |
|                    |                                                  |
|                    +--> mission_kg_builder.py -> mission_kg/          |
|                         (FIBO decomposition)     general/*.ttl       |
|                                                  ontology/*.ttl      |
|                                                                       |
+-------------- RAG Pipeline (rag_engine.py) --------------------------+
|                                                                       |
|  +--------------+   +----------------+   +--------------------+       |
|  | Vector Path  |   |  Graph Path    |   | Mission Evaluator  |       |
|  |              |   |                |   |                    |       |
|  | MiniLM-L6-v2 |   | rdflib TTL     |   | 8-Dimension Score  |       |
|  | FAISS Index  |   | NetworkX       |   | Tier/Confidence    |       |
|  | Chunk Search |   | Subgraph Ext.  |   | Evidence Linking   |       |
|  +--------------+   +----------------+   +--------------------+       |
|                                                                       |
+-------------- Streamlit App (app.py) --------------------------------+
|                                                                       |
|  Tab 1: Mission Statement      Tab 3: Portfolio Comparison           |
|    - Mission display + badge     - Heatmap (91 companies x 8 dims)   |
|    - 8-dim quick scores          - Sector comparison table           |
|    - Highlighted 10-K evidence   - Top/Bottom performers             |
|    - Source file + tier/conf     - Head-to-head comparison           |
|    - Raw text viewer             - CSV export                        |
|                                                                       |
|  Tab 2: Evidence-Based Reasoning  Tab 4: Knowledge Graph             |
|    - Detailed dim evaluation      - Schema dropdown:                 |
|    - Pattern analysis               Existing KG (Full)               |
|    - Buzzword detection              LLM-Driven (General)            |
|    - Stakeholder gaps                Ontology-Driven (FIBO)          |
|    - Semantic completeness         - PyVis interactive graph         |
|                                    - FIBO decomposition view         |
|                                                                       |
+-----------------------------------------------------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/AKSB0567/MSG-KG.git
cd MSG-KG
git checkout v3-mission-extraction-pipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Optional: HuggingFace Token
Only required for the LLM-based Q&A feature. Create a `.env` file:
```ini
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## Usage

### 1. Run the Mission Extraction Pipeline (first time)
```bash
# Extract mission statements from all 91 10-K filings
python mission_extractor.py

# Generate mission-focused knowledge graphs (General + FIBO)
python mission_kg_builder.py

# Build/update the company registry (optional, if KG data changes)
python build_registry.py
```

### 2. Launch the Dashboard
```bash
streamlit run app.py
```
Access at `http://localhost:8501`.

### 3. Navigating the Interface

**Left Panel: Company Selection**
- **Sector Filter**: Filter by industry (e.g., `6022 - State Commercial Banks`)
- **Company Selector**: Choose from 91 companies
- **Metadata**: Filing date, fiscal year, sector code

**Tab 1: Mission Statement**
- Mission display with overall quality badge (Outstanding/Strong/Adequate/Weak/Missing)
- 8-dimension quick score cards
- Source file name, extraction tier (Explicit/Implied/Inferred), confidence %
- Highlighted evidence chunk from 10-K with keyword matching
- Additional supporting passages
- Expandable raw 10-K source text viewer

**Tab 2: Evidence-Based Reasoning**
- Detailed evaluation per dimension with color-coded ratings
- Pattern analysis: buzzword detection, stakeholder coverage, semantic completeness
- Missing semantic components: Purpose, Activity, Audience, Differentiator, Domain

**Tab 3: Portfolio Comparison**
- Interactive Plotly heatmap (91 companies x 8 dimensions, 0-4 scale)
- Sort by any dimension or overall score
- Sector comparison table (average scores per industry)
- Top 5 / Bottom 5 performers
- Head-to-head company comparison table
- CSV download button for external analysis

**Tab 4: Knowledge Graph**
- **Schema dropdown** with 3 options:
  - *Existing KG (Full)*: Original SEC ontology graph (mission-focused subgraph)
  - *Mission KG: LLM-Driven*: Simple Company -> Mission graph
  - *Mission KG: Ontology-Driven*: FIBO-aligned decomposition with color-coded node types
- Interactive PyVis graph with legend, tooltips, zoom/pan
- FIBO decomposition breakdown: Stakeholders, Values, Objectives, Capabilities, Domains

---

## Project Structure

```text
MSG-KG/
|-- app.py                      # Streamlit dashboard (4 tabs, mission-focused)
|-- rag_engine.py               # Hybrid Graph-RAG engine (Vector + KG retrieval)
|-- mission_extractor.py        # Mission extraction pipeline (keyword scoring)
|-- mission_kg_builder.py       # Mission KG generator (General + FIBO schemas)
|-- build_registry.py           # Company registry builder (pre-computation)
|-- sic_codes.py                # SIC code-to-name mapping (200+ codes)
|-- kg_viewer.py                # Legacy interactive KG viewer (PyVis)
|-- companies_registry.json     # Pre-computed registry (91 companies)
|-- requirements.txt            # Python dependencies
|-- .env                        # API keys (not committed)
|
|-- MSGKG/                      # Original SEC Knowledge Graph data
|   |-- data/
|   |   |-- data/               # Cleaned 10-K text files (91 files, 3-23MB)
|   |   `-- output/             # TTL KG files (91 companies + schema.ttl)
|   |-- schema_1.owl            # OWL ontology (imports FIBO namespaces)
|   |-- nodes.csv               # Legacy graph nodes
|   `-- edges.csv               # Legacy graph edges
|
|-- data/
|   |-- mission_kg/
|   |   |-- general/            # LLM-Driven KGs (91 TTL files)
|   |   `-- ontology/           # FIBO-Aligned KGs (91 TTL files)
|   |-- chunks/                 # Semantic chunks for vector search
|   `-- item1/                  # Extracted Item 1 texts
|
`-- assets/
    `-- custom.css              # Corporate theme (Deep Red/Gold)
```

---

## Mission Extraction Pipeline

### How It Works

The `mission_extractor.py` pipeline processes each 10-K filing in 4 stages:

**Stage 1: SEC Header Stripping**
- Removes SGML headers (`<SEC-HEADER>...</SEC-HEADER>`)
- Skips 10-K reference lines and cover page boilerplate
- Finds `PART I` marker as the real content start

**Stage 2: Item 1 Section Extraction**
- Regex-based section detection: `ITEM 1. BUSINESS` to `ITEM 1A` or `ITEM 2`
- Fallback: first 100K characters after header stripping
- Cap at 200K characters for safety

**Stage 3: Sliding Window Keyword Scoring**
- Window: 600 characters, step: 250
- **Tier 1 patterns** (score 0.9-1.0): `"our mission is to"`, `"our purpose is"`, `"our vision is"`
- **Tier 2 keywords** (score 0.65-0.8): `"we are committed to"`, `"we strive to"`, `"we aim to"`
- **Tier 3 keywords** (score 0.4-0.5): `"is a leading"`, `"is a global"`, `"we provide"`
- **Boilerplate filter**: Skips windows with SEC/filing language
- **HCM deprioritization**: Reduces score by 0.35 for matches in HR/workforce sections
- **Position bonus**: Earlier text gets a small score boost

**Stage 4: Best Candidate Selection**
- Sort by tier (ascending), then by score (descending), then by position
- Clean extracted text (remove section markers, normalize whitespace)
- Store: mission text, tier, confidence, evidence chunk, keyword match

### Results

| Metric | Count |
|--------|-------|
| Total companies | 91 |
| From KG (TTL) | 11 |
| Tier 1 (Explicit) | 22 |
| Tier 2 (Implied) | 38 |
| Tier 3 (Inferred) | 16 |
| No mission found | 4 |
| **Total with mission** | **87 (95.6%)** |

Companies without missions: AIG, CNA Financial, Comerica, Union Carbide (genuinely absent from 10-K text).

---

## Knowledge Graph Schemas

### General Schema (LLM-Driven)

Simple flat graph per company:
```turtle
inst:Company_AMD a msg:Company ;
    rdfs:label "ADVANCED MICRO DEVICES INC" ;
    msg:hasMission inst:Mission_AMD .

inst:Mission_AMD a msg:MissionStatement ;
    msg:missionText "Build great products..." ;
    msg:missionTier 1 ;
    msg:missionConfidence 1.0 ;
    msg:extractedFrom "cleaned_10-K_0000002488-25-000012.txt" .
```

### Ontology Schema (FIBO-Aligned)

Decomposed mission graph with FIBO-inspired classes:
```turtle
@prefix fibo-fnd: <https://spec.edmcouncil.org/fibo/ontology/FND/> .
@prefix fibo-be: <https://spec.edmcouncil.org/fibo/ontology/BE/> .

inst:Mission_AMD a msg:MissionStatement ;
    msg:missionText "Build great products..." ;
    msg:servesStakeholder inst:AMD_Stakeholder_Customers ;
    msg:embodiesValue inst:AMD_Value_Innovation ;
    msg:pursuesObjective inst:AMD_Obj_AccelerateComputing ;
    msg:operatesInDomain inst:AMD_Domain_Technology .
```

**Decomposition categories:**
| Category | Description | Extraction Method |
|----------|-------------|-------------------|
| Stakeholders | Customer, Employee, Shareholder, Society, Partner, Government | Keyword matching (6 groups) |
| Values | Innovation, Sustainability, Integrity, Excellence, Safety, Inclusion, Growth, Service | Keyword matching (8 categories) |
| Objectives | Action-verb phrases from mission | Regex extraction |
| Domains | Business domain/industry | Mission + sector text matching |
| Capabilities | Core competencies | From KG registry data |

---

## Mission Evaluation Engine

### 8 Quality Dimensions (0-4 Scale)

| Dimension | What It Measures | Scoring Logic |
|-----------|-----------------|---------------|
| **Clarity of Purpose** | Clear "why we exist" statement | Sentence structure, purpose keywords, specificity |
| **Stakeholder Focus** | Who the company serves | Count of stakeholder groups mentioned |
| **Value Proposition** | What unique value is offered | Action verbs + differentiator language |
| **Actionability** | Concrete vs. abstract language | Action keywords, measurable terms |
| **Innovation Signal** | Forward-looking language | Innovation/technology keywords minus buzzwords |
| **Semantic Completeness** | Coverage of Purpose/Activity/Audience/How/Where | 5-component checklist |
| **Internal Consistency** | No contradictions or conflicts | Tonal consistency, length appropriateness |
| **Writing Quality** | Professional expression | Sentence length, passive voice, readability |

### Rating Scale

| Score | Rating | Color |
|-------|--------|-------|
| 4 | Outstanding | Dark Green |
| 3 | Strong | Green |
| 2 | Adequate | Yellow |
| 1 | Weak | Orange |
| 0 | Missing | Red |

---

## Deployment

### Streamlit Cloud

1. Connect GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/)
2. Deploy: Repository `AKSB0567/MSG-KG`, branch `v3-mission-extraction-pipeline`, main file `app.py`
3. Optional secrets (for LLM Q&A only):
   ```toml
   HUGGINGFACE_TOKEN = "your_token_here"
   ```

### Local
```bash
streamlit run app.py
# Access at http://localhost:8501
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit 1.30+ |
| **KG Storage** | RDF Turtle (.ttl) via rdflib |
| **Graph Engine** | NetworkX (DiGraph) |
| **Visualization** | Plotly (heatmaps) + PyVis (KG graphs) |
| **Mission Extraction** | Rule-based keyword scoring + SEC section parsing |
| **Mission KG** | FIBO-aligned ontology decomposition |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Reranker** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| **LLM** | Mixtral-8x7B via HuggingFace (optional, for Q&A) |
| **SIC Mapping** | Custom 200+ code dictionary |

---

## License
Copyright (c) 2026 MSG-KG Contributors. All Rights Reserved.
Finance Research — Confidential.
