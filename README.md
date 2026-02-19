# MSG-KG: Portfolio Intelligence System 🧠📊

**[🚀 MSG-KG Live Demo](https://msg-kg.streamlit.app/)**

**Mission-Strategy-Goals Knowledge Graph (MSG-KG)** is a financial intelligence platform that combines **Retrieval-Augmented Generation (RAG)** with structured **Knowledge Graphs (KG)** to analyze corporate strategy alignment.

Built for **Finance Research**, this system ingests SEC 10-K filings, constructs a graph of company missions, strategies, and risks, and provides an interactive dashboard for comparative analysis.

---

## 🚀 Key Features

### 1. 🏗️ Hybrid RAG Engine
- **Dual-Path Retrieval**: Combines unstructured vector search (FAISS + FinBERT) with structured graph traversal (NetworkX).
- **Cross-Encoder Reranking**: Re-scores results using `ms-marco-MiniLM` for high precision.
- **Explainable AI**: Returns not just answers but **evidence spans** (text chunks) and **graph paths** (reasoning chains).

### 2. 🕸️ Knowledge Graph Construction
- **Automated Extraction**: Uses Large Language Models (Mistral/Mixtral) to extract entities:
  - *Mission, Vision, Strategic Objectives, Capabilities, Initiatives, Risks*.
- **Rule-Based Fallback**: robust regex patterns ensuring data availability even without LLM.
- **Interactive Visualization**: Native `PyVis` integration for exploring company connections.

### 3. ⚖️ Comparative Analytics
- **Strategic Alignment Scorecard**: Quantitative assessment of Mission-to-Execution consistency.
- **Portfolio Heatmap**: Plotly-based interactive visualization of alignment scores across the portfolio.
- **Head-to-Head**: Direct side-by-side comparison of company strategies.

### 4. 🎨 Professional UI
- ** Layout**: Persistent left filter panel for context-aware navigation.
- **Enterprise Theming**: Deep Red/Gold branding (`#CD1515`) inspired by top-tier portals.
- **Contextual Controls**: Filters appear alongside strictly relevant content.

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- [HuggingFace Account](https://huggingface.co/) (for LLM inference)

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/MSG-KG.git
   cd MSG-KG
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```ini
   HUGGINGFACE_TOKEN=your_hf_token_here
   ```

---

## 🚦 Usage

### 1. Run the Dashboard
```bash
streamlit run app.py
```
Access the app at `http://localhost:8501`.

### 2. Navigating the Interface
The application uses an **style 2-column layout**:

#### 🏗️ Left Filter Panel
This dedicated column contains all context controls:
- **Portfolio Context**: Switch between "All", "Good" (High Alignment), or "Bad" (Low Alignment) portfolios.
- **Company Selector**: Choose the target company for analysis.
- **RAG Model Config**: Toggle "Rerank" and "Explain Mode" to control the depth of analysis.

#### 📊 Dashboard Tabs (Right Panel)
1. **Overview**: High-level financial summary, Mission card, and KG statistics.
2. **Ask KG-RAG**: Chat interface to query the system (e.g., *"How does Apple's privacy strategy impact revenue?"*).
3. **Competitive Landscape**: Heatmaps and Scorecards comparing companies.
4. **Evidence Explorer**: Inspect raw 10-K text chunks and confidence scores.
5. **KG View**: Manipulate the interactive Knowledge Graph node-link diagram.

---

## 📂 Project Structure

```text
MSG-KG/
├── app.py                 # Main Streamlit application entry point
├── rag_engine.py          # Core RAG logic (Vector + Graph retrieval)
├── kg_builder.py          # Knowledge Graph construction pipeline
├── sec_extractor.py       # SEC 10-K crawling and parsing module
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not committed)
├── MSGKG/                 # Knowledge Graph storage
│   ├── nodes.csv          # Graph nodes (Entities)
│   ├── edges.csv          # Graph edges (Relationships)
│   └── *_kg.json          # Raw per-company extraction data
├── data/
│   ├── item1/             # Raw text from SEC filings
│   └── chunks/            # Chunked JSONs for vector indexing
└── assets/
    └── custom.css         # Enterprise styling stylesheet
```

---

## 🧠 Methodology

### Data Pipeline
1. **Extraction**: `sec_extractor.py` pulls Item 1 (Business) from EDGAR.
2. **Chunking**: Text is split into 512-token overlapping chunks.
3. **Vectorization**: Chunks are embedded using `FinBERT` and indexed in `FAISS`.
4. **Graph Building**: `kg_builder.py` extracts entities/relations to build `NetworkX` graph.

### Query Pipeline
When a user asks a question:
1. **Vector Search**: Retrieves top-k relevant text chunks.
2. **Graph Search**: Traverses edges from the Company node to find relevant concepts.
3. **Fusion**: Reranker scores combined results.
4. **Generation**: LLM synthesizes an answer citing both *Text Evidence* and *Graph Paths*.

---

## 🌐 Live Deployment (Streamlit Cloud)

To deploy this dashboard to a public URL:
1.  **Authorize**: Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
2.  **Deploy**: Select this repository and `app.py` as the main entry point.
3.  **Secrets**: In the Streamlit Cloud dashboard, go to **Settings > Secrets** and paste your token:
    ```toml
    HUGGINGFACE_TOKEN = "your_token_here"
    ```

---

## 📜 License
Copyright (c) 2026 MSG-KG Contributors. All Rights Reserved.
Privileged & Confidential — Enterprise Internal Use Only.
