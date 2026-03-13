---
title: MSG-KG
emoji: 📊
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: "1.31.1"
app_file: app.py
pinned: false
---

# MSG-KG: Mission Statement Knowledge Graph Intelligence

**Mission-Strategy-Goals Knowledge Graph (MSG-KG)** is a financial intelligence platform that extracts, validates, and compares corporate mission statements from **91 S&P company SEC 10-K filings** using a **dual-pipeline approach** — rule-based text matching and **LLM-driven semantic validation** (Qwen2.5-72B-Instruct) — with **FIBO ontology-aligned Knowledge Graphs** and portfolio-level comparative analytics.

Built for **Finance Research**, this system provides evidence-based mission statement analysis with interactive visualizations, multi-schema knowledge graphs, a 4-phase mission improvement pipeline, and a live toggle between extraction approaches.

---

## Table of Contents

- [Key Features](#key-features)
- [Dual Validation Pipeline](#dual-validation-pipeline)
- [Mission Improvement Pipeline](#mission-improvement-pipeline)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Mission Extraction Pipeline](#mission-extraction-pipeline)
- [LLM Validation Pipeline](#llm-validation-pipeline)
- [Knowledge Graph Schemas](#knowledge-graph-schemas)
- [Mission Evaluation Engine](#mission-evaluation-engine)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Key Features

### 1. Dual Validation Pipeline (Text Matching + LLM)
- **Sidebar toggle** switches between two independent validation approaches across ALL tabs
- **Text Matching Pipeline**: Original keyword-based extraction + back-verification against source filing
- **LLM-Driven Pipeline**: Qwen2.5-72B-Instruct semantic validation + 4-phase mission improvement
- Metrics, heatmaps, and evaluations update dynamically with the selected approach
- Before/after comparison expanders for improved companies

### 2. Mission Statement Extraction Pipeline
- **91 companies** processed from SEC 10-K filings
- **3-tier classification**: Explicit (Tier 1), Implied (Tier 2), Inferred (Tier 3)
- SEC SGML header stripping + Item 1 section extraction
- Sliding window keyword scoring with HCM section deprioritization
- Confidence scoring and evidence provenance tracking

### 3. LLM-Driven Mission Validation & Improvement
- **Qwen2.5-72B-Instruct** via HuggingFace Inference API (Novita provider)
- Semantic classification: YES / PARTIAL / NO verdict for each mission
- Content type detection: MISSION, VISION, STRATEGY, OPERATIONS, HR, FINANCIAL, MIXED, REGULATORY
- **5-level mission quality classification** (M0-M4):
  - M0: Absent from filing
  - M1: Directly stated as mission/purpose
  - M2: Stated but requires inference
  - M3: Not stated, derived from context
  - M4: Present but vague
- **4-phase improvement pipeline** with before/after tracking (see below)

### 4. Dual-Schema Mission Knowledge Graphs
- **General Schema (LLM-Driven)**: Flat Company -> Mission graph per company
- **Ontology Schema (FIBO-Aligned)**: Mission decomposed into Stakeholders, Values, Objectives, Capabilities, and Business Domains
- **182 TTL files** (91 general + 91 ontology) with 280 total triplets
- Interactive PyVis visualization with schema switching dropdown

### 5. 4-Dimension Mission Quality Evaluation
- **Clarity of Purpose** | **Stakeholder Focus** | **Value Proposition** | **Actionability**
- 0-4 scale: Outstanding / Strong / Adequate / Weak / Missing
- Buzzword detection, stakeholder gap analysis, semantic component analysis
- Scores update dynamically based on active pipeline toggle

### 6. Evidence-Based Reasoning
- Highlighted evidence from actual 10-K text files matched by CIK
- Source file name, extraction tier, confidence score displayed
- Keyword-density matching to find mission passages in 3-23MB documents
- Raw source text viewer with contextual highlighting
- LLM assessment, confidence %, content type classification, and issue flagging

### 7. Portfolio Comparison Analytics
- Interactive heatmap across all 91 companies and 4 dimensions
- Sector comparison (average scores per industry)
- Top/bottom performer rankings
- Head-to-head company comparison
- CSV export for external analysis

### 8. SIC Code Normalization
- 200+ SIC code mappings (e.g., `6022` -> `State Commercial Banks`)
- All sectors displayed as `Code - Industry Name` format
- Sector-based filtering in the dashboard

---

## Dual Validation Pipeline

The dashboard provides a **sidebar radio toggle** to switch between two independent approaches. All tabs (Mission Statement, Evidence-Based Reasoning, Portfolio Comparison, Knowledge Graph) respond to this toggle.

### Text Matching (Original Extraction)
- Mission extracted via sliding-window keyword scoring
- Back-verified against 10-K source using exact string matching + keyword density
- Shows: verification status, match type, evidence highlighting, raw source viewer
- Filters: Verification Status (Verified/Partial/Not Found)

### LLM-Driven (Qwen2.5-72B Improved)
- Same extraction + LLM semantic validation + 4-phase improvement
- Shows: LLM verdict (YES/PARTIAL/NO), content type, quality code, confidence %, assessment
- For improved companies: before/after expander showing original vs. improved mission
- For M0 companies: business purpose description (purple styling)
- Filters: LLM Verdict (YES/PARTIAL/NO), Content Type (MISSION/VISION/STRATEGY/etc.)

### Example: What Changes Between Pipelines

| Company | Text Matching (Original) | LLM Pipeline (Improved) | Change |
|---------|--------------------------|-------------------------|--------|
| **American Electric Power** | "We are committed to fundamentally embed layers of protection in the work we do." | "AEP is committed to providing reliable affordable power to its customers." | Re-extracted (was HR language, not mission) |
| **American Express** | "We aim to provide the world's best customer experience every day and our reputation for world-class service has been recognized by numerous awards..." | "We aim to provide the world's best customer experience every day." | Trimmed (removed non-mission trailing text) |
| **Astronics Corp** | "Increase value by developing technologies internally or through acquisition for targeted markets; ASTRONICS CORP -- Mission." | "Increase value by developing technologies and capabilities, either internally or through acquisition, and using those capabilities to provide innovative solutions to our targeted markets..." | Expanded (full sentence from source) |
| **American Airlines** | "We are committed to engaging with our stakeholders to seek to advance these initiatives and have de." (truncated) | Business Purpose: "...operates a major network air carrier, providing scheduled air transportation for passengers and cargo to over 350 destinations worldwide..." | Replaced with honest business purpose (no mission in filing) |
| **1st Source Corp** | "Help clients achieve security, build wealth, and realize their dreams." | Same (unchanged) | Validated as genuine mission (YES verdict) |

---

## Mission Improvement Pipeline

The `mission_improver.py` script runs 4 phases to produce the best possible mission statement for each company, with full before/after tracking:

### Phase 1: Apply PARTIAL Corrections (31 companies)
- Companies where LLM verdict was PARTIAL had `llm_corrected_mission` applied
- Example: trailing non-mission text trimmed, section markers removed
- Saves `mission_before_improvement` for comparison

### Phase 2: Re-extract from Source (12 companies)
- Companies with LLM verdict NO re-processed against full Item 1 text
- LLM re-reads up to 80K characters of source filing to find the real mission
- Multiple search strategies: beginning of text + keyword-anchored passages
- Tracks `reextract_source_type` and `reextract_confidence`

### Phase 3: Business Purpose Fallback (34 companies)
- Remaining NO companies get a `business_purpose` field (what the company does and who it serves)
- Honest labeling: displayed as "Business Purpose" (purple styling), not a mission statement
- Separate from `mission` field to maintain data integrity

### Phase 4: Re-validate Changed Missions (37 companies)
- All companies modified in Phases 1-2 re-validated by LLM
- Quality codes (M0-M4) updated based on new content
- Confidence scores refreshed

### Results Summary

| Category | Count |
|----------|-------|
| LLM Verdict: YES (genuine mission) | 13 |
| LLM Verdict: PARTIAL (corrected) | 31 |
| LLM Verdict: NO (re-extracted or business purpose) | 47 |
| Missions improved (before != after) | 43 |
| Business purpose added | 34 |
| Total companies | 91 |

---

## End-to-End Pipeline

The system processes SEC 10-K filings through 8 stages:

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

### Stage 3: Mission Statement Extraction (Text Matching)
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
- **Ontology Schema**: FIBO-aligned decomposition using rule-based NLP

### Stage 5: LLM Validation
```
companies_registry.json -> llm_validate_missions.py -> companies_registry.json (enriched)
```
- Qwen2.5-72B-Instruct evaluates each extracted mission semantically
- Adds: `llm_is_mission`, `llm_actual_type`, `llm_quality`, `llm_confidence`, `llm_corrected_mission`, `llm_issues`
- Reclassifies mission quality codes (M0-M4) based on LLM assessment

### Stage 6: Mission Improvement (4 Phases)
```
companies_registry.json -> mission_improver.py -> companies_registry.json (improved)
```
- Phase 1: Apply PARTIAL corrections (31 companies)
- Phase 2: Re-extract from source for NO cases (12 companies)
- Phase 3: Business purpose fallback for remaining NO cases (34 companies)
- Phase 4: Re-validate all changed missions
- Preserves `mission_before_improvement` for before/after comparison

### Stage 7: Registry Building
```
*.ttl + cleaned_10-K_*.txt -> build_registry.py -> companies_registry.json
```
- Pre-computes company metadata, KG stats, alignment scores
- Mission fallback chain: KG -> Item 1 text -> cleaned 10-K text
- SIC code normalization via `sic_codes.py`
- Output: 91 companies with mission, sector, scores, evidence metadata

### Stage 8: Interactive Dashboard
```
companies_registry.json + *.ttl -> app.py (Streamlit) -> http://localhost:8501
```
- 4-tab dashboard: Mission Statement | Evidence-Based Reasoning | Portfolio Comparison | Knowledge Graph
- Dual-pipeline toggle (Text Matching vs LLM-Driven) in sidebar
- Real-time mission evaluation (4 dimensions, 0-4 scale)
- KG schema switching (Existing / LLM-Driven / Ontology-Driven)
- Evidence highlighting from 10-K source files

---

## Architecture

```
+-------------------------------------------------------------------------+
|                    MSG-KG Architecture v4.0                               |
+-------------------------------------------------------------------------+
|                                                                           |
|  SEC 10-K Filing --+--> sec_extractor.py --------> cleaned_10-K_*.txt    |
|                    |                                                      |
|                    +--> kg_builder.py ------------> *.ttl (SEC KG)        |
|                    |                                                      |
|                    +--> mission_extractor.py ------> companies_registry   |
|                    |    (keyword scoring,             .json               |
|                    |     tier classification)                              |
|                    |                                                      |
|                    +--> mission_kg_builder.py -----> mission_kg/          |
|                    |    (FIBO decomposition)          general/*.ttl       |
|                    |                                  ontology/*.ttl      |
|                    |                                                      |
|                    +--> llm_validate_missions.py --> registry + LLM       |
|                    |    (Qwen2.5-72B-Instruct)       validation fields    |
|                    |                                                      |
|                    +--> mission_improver.py -------> registry + improved  |
|                         (4-phase improvement)        missions + biz_purp  |
|                                                                           |
+-------------- RAG Pipeline (rag_engine.py) ------------------------------+
|                                                                           |
|  +--------------+   +----------------+   +--------------------+           |
|  | Vector Path  |   |  Graph Path    |   | Mission Evaluator  |           |
|  |              |   |                |   |                    |           |
|  | MiniLM-L6-v2 |   | rdflib TTL     |   | 4-Dimension Score  |           |
|  | FAISS Index  |   | NetworkX       |   | Tier/Confidence    |           |
|  | Chunk Search |   | Subgraph Ext.  |   | Evidence Linking   |           |
|  +--------------+   +----------------+   +--------------------+           |
|                                                                           |
+-------------- Streamlit App (app.py) ------------------------------------+
|                                                                           |
|  [Sidebar: Pipeline Toggle + Dynamic Filters]                            |
|    - Text Matching / LLM-Driven radio button                             |
|    - Sector filter, company selector, metadata card                      |
|    - Dynamic filters: Verification Status OR LLM Verdict/Content Type    |
|                                                                           |
|  Tab 1: Mission Statement       Tab 3: Portfolio Comparison              |
|    - Mission box (styled HTML)    - Heatmap (91 cos x 4 dims)            |
|    - Quality badge (M0-M4)       - Sector comparison table              |
|    - 4-dim quick score cards     - Top/Bottom performers                |
|    - Before/after expander       - Head-to-head comparison              |
|    - Validation evidence          - CSV export                           |
|    - LLM assessment/issues                                               |
|                                                                           |
|  Tab 2: Evidence-Based Reasoning  Tab 4: Knowledge Graph                 |
|    - Detailed dim evaluation       - Schema dropdown:                    |
|    - Pattern analysis                Existing KG (Full)                  |
|    - Buzzword detection              LLM-Driven (General)               |
|    - Stakeholder gaps                Ontology-Driven (FIBO)             |
|    - Semantic completeness         - PyVis interactive graph             |
|                                    - FIBO decomposition view             |
|                                                                           |
+--------------------------------------------------------------------------+
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
git checkout v2-kg-rag-pipeline

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### HuggingFace Token
Required for LLM validation and Q&A features. Create a `.env` file:
```ini
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## Usage

### 1. Run the Full Pipeline (first time)
```bash
# Extract mission statements from all 91 10-K filings
python mission_extractor.py

# Generate mission-focused knowledge graphs (General + FIBO)
python mission_kg_builder.py

# Run LLM validation (requires HF token)
python llm_validate_missions.py

# Run 4-phase mission improvement
python mission_improver.py

# Build/update the company registry
python build_registry.py
```

### 2. Launch the Dashboard
```bash
streamlit run app.py
```
Access at `http://localhost:8501`.

### 3. Navigating the Interface

**Sidebar: Pipeline Control**
- **Validation Approach**: Toggle between Text Matching and LLM-Driven
- **Sector Filter**: Filter by industry (e.g., `6022 - State Commercial Banks`)
- **Company Selector**: Choose from 91 companies
- **Dynamic Filters**: Verification status (text matching) OR LLM verdict/content type (LLM mode)
- **Metadata Card**: Filing date, fiscal year, sector code, version badge (Original/Improved/Business Purpose)

**Tab 1: Mission Statement**
- Mission display with overall quality badge (Outstanding/Strong/Adequate/Weak/Missing)
- Mission quality classification badge (M0-M4)
- 4-dimension quick score cards
- Before/after comparison expander for improved companies
- **Text Matching mode**: verification status, match type, evidence highlighting, raw source viewer, additional supporting passages
- **LLM mode**: LLM verdict (YES/PARTIAL/NO), content type, confidence %, assessment, issues, suggested mission

**Tab 2: Evidence-Based Reasoning**
- Detailed evaluation per dimension with color-coded ratings
- Pattern analysis: buzzword detection, stakeholder coverage, semantic completeness
- Missing semantic components: Purpose, Activity, Audience, Differentiator, Domain

**Tab 3: Portfolio Comparison**
- Interactive Plotly heatmap (91 companies x 4 dimensions, 0-4 scale)
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
|-- app.py                      # Streamlit dashboard (4 tabs, dual-pipeline toggle)
|-- rag_engine.py               # Hybrid Graph-RAG engine (Vector + KG retrieval)
|-- mission_extractor.py        # Mission extraction pipeline (keyword scoring)
|-- mission_kg_builder.py       # Mission KG generator (General + FIBO schemas)
|-- llm_validate_missions.py    # LLM validation (Qwen2.5-72B-Instruct)
|-- mission_improver.py         # 4-phase mission improvement pipeline
|-- build_registry.py           # Company registry builder (pre-computation)
|-- sic_codes.py                # SIC code-to-name mapping (200+ codes)
|-- kg_viewer.py                # Legacy interactive KG viewer (PyVis)
|-- companies_registry.json     # Pre-computed registry (91 companies, enriched)
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

### Text Matching Results

| Metric | Count |
|--------|-------|
| Total companies | 91 |
| From KG (TTL) | 11 |
| Tier 1 (Explicit) | 22 |
| Tier 2 (Implied) | 38 |
| Tier 3 (Inferred) | 16 |
| No mission found | 4 |
| **Total with mission** | **87 (95.6%)** |

---

## LLM Validation Pipeline

### How It Works

The `llm_validate_missions.py` script sends each extracted mission to **Qwen2.5-72B-Instruct** via the HuggingFace Inference API (Novita provider) for semantic evaluation.

**For each company, the LLM determines:**
1. **Is this a mission statement?** (YES / PARTIAL / NO)
2. **What type of content is it?** (MISSION, VISION, STRATEGY, OPERATIONS, HR, FINANCIAL, MIXED, REGULATORY)
3. **Quality classification** (M0-M4)
4. **Confidence score** (0-100%)
5. **Issues found** (e.g., "truncated", "contains_operations", "hr_language")
6. **Corrected mission** (for PARTIAL cases)

### LLM Validation Results

| Verdict | Count | Description |
|---------|-------|-------------|
| YES | 13 | Genuine mission statement, no changes needed |
| PARTIAL | 31 | Contains mission but needs correction (trimming, cleaning) |
| NO | 47 | Not a mission statement (strategy, operations, HR, etc.) |

### Mission Quality Classification (M0-M4)

| Code | Label | Description |
|------|-------|-------------|
| M0 | Absent | No mission statement in the filing |
| M1 | Direct | Explicitly stated as mission/purpose/vision |
| M2 | Inference | Mission-like statement requiring interpretation |
| M3 | Context | Derived from business description, not stated directly |
| M4 | Vague | Present but generic corporate language |

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

### 4 Quality Dimensions (0-4 Scale)

| Dimension | What It Measures | Scoring Logic |
|-----------|-----------------|---------------|
| **Clarity of Purpose** | Clear "why we exist" statement | Sentence structure, purpose keywords, specificity |
| **Stakeholder Focus** | Who the company serves | Count of stakeholder groups mentioned |
| **Value Proposition** | What unique value is offered | Action verbs + differentiator language |
| **Actionability** | Concrete vs. abstract language | Action keywords, measurable terms |

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
2. Deploy: Repository `AKSB0567/MSG-KG`, branch `v2-kg-rag-pipeline`, main file `app.py`
3. Secrets (for LLM features):
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
| **Frontend** | Streamlit 1.31+ |
| **KG Storage** | RDF Turtle (.ttl) via rdflib |
| **Graph Engine** | NetworkX (DiGraph) |
| **Visualization** | Plotly (heatmaps) + PyVis (KG graphs) |
| **Mission Extraction** | Rule-based keyword scoring + SEC section parsing |
| **LLM Validation** | Qwen2.5-72B-Instruct via HuggingFace Inference API (Novita provider) |
| **Mission Improvement** | 4-phase pipeline (PARTIAL correction, re-extraction, business purpose, re-validation) |
| **Mission KG** | FIBO-aligned ontology decomposition |
| **Embeddings** | all-MiniLM-L6-v2 (sentence-transformers) |
| **Reranker** | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| **LLM (KG Build)** | Mixtral-8x7B via HuggingFace (for KG entity extraction) |
| **SIC Mapping** | Custom 200+ code dictionary |

---

## License
Copyright (c) 2026 MSG-KG Contributors. All Rights Reserved.
Finance Research — Confidential.
