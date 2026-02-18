"""
RAG Engine: Hybrid Vector + Graph Retrieval
===========================================
Core retrieval logic for the MSG-KG system. This module implements a hybrid 
approach, combining semantic vector search with structured graph traversal.

Architectural Flow:
1. Vector Retrieval: FAISS + FinBERT semantic search on SEC chunks.
2. Graph Retrieval: NetworkX exploration to find strategic relations.
3. Reranking: Cross-encoder re-scoring of combined evidence.
4. LLM synthesis: Generates expert-level financial insights.
"""

import os, json, pathlib, re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = pathlib.Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
KG_DIR     = BASE_DIR / "MSGKG"

# ── Lazy-loaded globals ──────────────────────────────────────────────────
_embedder = None
_faiss_index = None
_all_chunks = None
_kg_graph = None


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class EvidenceSpan:
    doc_type: str = "10-K"
    section: str = "Item 1A"
    page: int = 0
    confidence: float = 0.0
    text: str = ""
    chunk_id: str = ""

@dataclass
class KGPath:
    path: str = ""
    description: str = ""
    weight: float = 1.0

@dataclass
class RAGAnswer:
    question: str = ""
    company: str = ""
    answer: str = ""
    evidence_spans: list = field(default_factory=list)
    kg_paths: list = field(default_factory=list)
    pipeline_steps: list = field(default_factory=list)


# ── Embedding Model (FinBERT) ───────────────────────────────────────────

def get_embedder():
    """Lazy-load FinBERT sentence transformer."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading FinBERT embedding model...")
            _embedder = SentenceTransformer("ProsusAI/finbert")
            print("✓ FinBERT loaded")
        except Exception as e:
            print(f"⚠ FinBERT failed ({e}), falling back to all-MiniLM-L6-v2")
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


# ── Vector Store (FAISS) ────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    """Load all chunk JSONs from data/chunks/."""
    global _all_chunks
    if _all_chunks is not None:
        return _all_chunks

    chunks = []
    if CHUNKS_DIR.exists():
        for fp in sorted(CHUNKS_DIR.glob("*_chunks.json")):
            with open(fp, encoding="utf-8") as f:
                chunks.extend(json.load(f))
    _all_chunks = chunks
    return chunks


def build_faiss_index(chunks: list[dict] = None):
    """Build FAISS index from chunk embeddings."""
    global _faiss_index, _all_chunks
    import faiss

    if chunks is None:
        chunks = load_all_chunks()

    if not chunks:
        print("⚠ No chunks to index")
        return None

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
    index.add(embeddings)

    _faiss_index = index
    _all_chunks = chunks
    print(f"✓ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def vector_search(query: str, company: str = None, top_k: int = 10) -> list[dict]:
    """Search FAISS index for most relevant chunks."""
    global _faiss_index, _all_chunks

    if _faiss_index is None:
        build_faiss_index()

    if _faiss_index is None or not _all_chunks:
        return []

    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    import faiss
    faiss.normalize_L2(q_emb)

    # Search more to allow company filtering
    search_k = min(top_k * 5, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q_emb, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = _all_chunks[idx].copy()
        chunk["score"] = float(score)

        # Filter by company if specified
        if company and chunk.get("ticker") != company:
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results


# ── Knowledge Graph ─────────────────────────────────────────────────────

def load_kg() -> nx.DiGraph:
    """Load KG from nodes.csv and edges.csv into NetworkX DiGraph."""
    global _kg_graph
    if _kg_graph is not None:
        return _kg_graph

    G = nx.DiGraph()
    nodes_path = KG_DIR / "nodes.csv"
    edges_path = KG_DIR / "edges.csv"

    if nodes_path.exists():
        df_nodes = pd.read_csv(nodes_path, encoding="utf-8")
        for _, row in df_nodes.iterrows():
            node_id = row[":ID"]
            attrs = {k: v for k, v in row.items()
                     if k != ":ID" and pd.notna(v) and str(v).strip()}
            G.add_node(node_id, **attrs)

    if edges_path.exists():
        df_edges = pd.read_csv(edges_path, encoding="utf-8")
        for _, row in df_edges.iterrows():
            G.add_edge(row[":START_ID"], row[":END_ID"],
                       relation=row[":TYPE"])

    _kg_graph = G
    print(f"✓ KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def graph_search(company: str, topic: str = None, top_k: int = 10) -> list[dict]:
    """
    Retrieve KG subgraph centered on a company.
    Returns nodes and paths relevant to the query topic.
    """
    G = load_kg()
    company_node = f"Company:{company}"

    if company_node not in G:
        return []

    results = []

    # Get all neighbors (1-hop)
    for neighbor in G.neighbors(company_node):
        edge_data = G[company_node][neighbor]
        node_data = G.nodes[neighbor]
        label = node_data.get(":LABEL", "Unknown")
        text = node_data.get("text", "") or node_data.get("name", "")

        # Basic topic relevance scoring
        score = 0.5
        if topic:
            topic_lower = topic.lower()
            text_lower = (text or "").lower()
            label_lower = label.lower()
            if topic_lower in text_lower:
                score = 0.9
            elif any(w in text_lower for w in topic_lower.split()):
                score = 0.7

        results.append({
            "node_id": neighbor,
            "label": label,
            "text": text,
            "relation": edge_data.get("relation", ""),
            "score": score,
        })

    # Get 2-hop paths
    for n1 in G.neighbors(company_node):
        for n2 in G.neighbors(n1):
            if n2 != company_node:
                n1_data = G.nodes[n1]
                n2_data = G.nodes[n2]
                e1 = G[company_node][n1].get("relation", "?")
                e2 = G[n1][n2].get("relation", "?")

                results.append({
                    "node_id": n2,
                    "label": n2_data.get(":LABEL", "Unknown"),
                    "text": n2_data.get("text", "") or n2_data.get("name", ""),
                    "relation": f"{e1} → {e2}",
                    "score": 0.4,
                    "path": f"Company:{company} →[{e1}]→ {n1_data.get(':LABEL','')} →[{e2}]→ {n2_data.get(':LABEL','')}",
                })

    # Sort by score and limit
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_kg_paths(company: str) -> list[KGPath]:
    """Extract meaningful KG paths for explain mode."""
    G = load_kg()
    company_node = f"Company:{company}"
    paths = []

    if company_node not in G:
        return paths

    # Find all paths of length 2-4 from company
    for n1 in G.neighbors(company_node):
        n1_label = G.nodes[n1].get(":LABEL", "")
        e1 = G[company_node][n1].get("relation", "")

        path_str = f"Company:{company} → {n1_label}"
        paths.append(KGPath(
            path=f"Company:{company} →[{e1}]→ {n1_label}",
            description=G.nodes[n1].get("text", "") or G.nodes[n1].get("name", ""),
        ))

        for n2 in G.neighbors(n1):
            if n2 != company_node:
                n2_label = G.nodes[n2].get(":LABEL", "")
                e2 = G[n1][n2].get("relation", "")
                paths.append(KGPath(
                    path=f"Company:{company} → {n1_label} → {n2_label}",
                    description=f"{e1} → {e2}",
                ))

                for n3 in G.neighbors(n2):
                    if n3 != company_node and n3 != n1:
                        n3_label = G.nodes[n3].get(":LABEL", "")
                        e3 = G[n2][n3].get("relation", "")
                        paths.append(KGPath(
                            path=f"Company:{company} → {n1_label} → {n2_label} → {n3_label}",
                            description=f"{e1} → {e2} → {e3}",
                        ))

    return paths


# ── Reranker ────────────────────────────────────────────────────────────

def rerank(query: str, results: list[dict], top_k: int = 10) -> list[dict]:
    """
    Cross-encoder reranking of combined results.
    Falls back to score-based ranking if cross-encoder unavailable.
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(query, r.get("text", "")) for r in results]
        scores = model.predict(pairs)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    except Exception:
        # Fallback: just sort by original score
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return results[:top_k]


# ── Curated company metadata for RAG prompt enrichment ──────────────────
COMPANY_META = {
    "AAPL": {
        "name": "Apple Inc.",
        "mission": "To bring the best user experience to customers through innovative hardware, software, and services.",
        "alignment_label": "Worst-Alignment",
    },
    "AMD": {
        "name": "Advanced Micro Devices",
        "mission": "To build great products that accelerate next-generation computing experiences through high-performance and adaptive computing technology.",
        "alignment_label": "Worst-Alignment",
    },
    "TGT": {
        "name": "Target Corporation",
        "mission": "To help all families discover the joy of everyday life by delivering an experience that is uniquely Target.",
        "alignment_label": "Worst-Alignment",
    },
    "WMT": {
        "name": "Walmart Inc.",
        "mission": "To help people save money and live better — through everyday low prices, powered by everyday low cost.",
        "alignment_label": "Best-Alignment",
    },
    "TSN": {
        "name": "Tyson Foods Inc.",
        "mission": "To raise the world's expectations for how much good food can do — feeding people sustainably, responsibly, and well.",
        "alignment_label": "Best-Alignment",
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "mission": "To empower every person and every organization on the planet to achieve more.",
        "alignment_label": "Best-Alignment",
    },
}


def generate_answer(question: str, company: str, context_chunks: list[dict],
                    kg_context: list[dict]) -> str:
    """Generate answer using HuggingFace LLM with MSG-KG alignment context."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Get curated company metadata
    meta = COMPANY_META.get(company, {})
    company_name = meta.get("name", company)
    mission = meta.get("mission", "N/A")
    alignment = meta.get("alignment_label", "N/A")

    # Build evidence context (more context, better formatted)
    context_parts = []
    for i, chunk in enumerate(context_chunks[:8]):
        score = chunk.get('score', 0) or chunk.get('rerank_score', 0)
        context_parts.append(
            f"[Evidence {i+1}] (10-K {chunk.get('section', 'Item 1')}, "
            f"Page ~{chunk.get('page_estimate', '?')}, Score: {score:.2f})\n"
            f"{chunk['text'][:600]}"
        )

    # Build KG context (structured by type)
    kg_by_type = {}
    for item in kg_context[:10]:
        label = item.get('label', 'Other')
        text = item.get('text', '')[:200]
        relation = item.get('relation', '')
        if label not in kg_by_type:
            kg_by_type[label] = []
        kg_by_type[label].append(f"{text} (via {relation})")

    kg_parts = []
    for label, items in kg_by_type.items():
        kg_parts.append(f"  [{label}]")
        for it in items[:3]:
            kg_parts.append(f"    - {it}")

    context_text = "\n\n".join(context_parts) if context_parts else "No 10-K evidence retrieved."
    kg_text = "\n".join(kg_parts) if kg_parts else "No KG context available."

    prompt = f"""<s>[INST] You are a financial analyst specializing in the MSG-KG (Mission-Strategy-Goals Knowledge Graph) framework. Your job is to analyze SEC 10-K filings and answer questions using both document evidence and knowledge graph data.

COMPANY PROFILE:
- Name: {company_name} ({company})
- Mission: {mission}
- Alignment Classification: {alignment}

10-K FILING EVIDENCE:
{context_text}

KNOWLEDGE GRAPH DATA:
{kg_text}

QUESTION: {question}

Instructions:
1. Ground your answer in the retrieved evidence — cite specific passages
2. Reference the knowledge graph where it strengthens your answer
3. Connect the answer back to the company's mission and strategic alignment when relevant
4. If evidence is insufficient, clearly say what is missing
5. Be concise but thorough — aim for 3-5 key insights

ANSWER: [/INST]"""

    if hf_token:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=hf_token)
            response = client.text_generation(
                prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_new_tokens=700,
                temperature=0.25,
                return_full_text=False,
            )
            answer = response.strip()
            # Clean up any trailing incomplete sentences
            if answer and not answer[-1] in '.!?"':
                last_period = answer.rfind('.')
                if last_period > len(answer) * 0.5:
                    answer = answer[:last_period + 1]
            return answer
        except Exception as e:
            print(f"⚠ LLM generation failed: {e}")

    # Fallback: structured extractive summary from top chunks + KG
    parts = []
    parts.append(f"**{company_name}** ({alignment})")
    parts.append(f"**Mission:** {mission}\n")

    if context_chunks:
        parts.append("**Key Findings from 10-K:**")
        for i, chunk in enumerate(context_chunks[:4]):
            text = chunk["text"][:250].strip()
            parts.append(f"{i+1}. {text}")

    if kg_context:
        parts.append("\n**Knowledge Graph Insights:**")
        for item in kg_context[:4]:
            parts.append(f"- **{item.get('label','')}**: {item.get('text','')[:150]}")

    return "\n\n".join(parts) if parts else f"No relevant information found for {company} regarding this question."


# ── Main RAG Pipeline ───────────────────────────────────────────────────

def ask(question: str, company: str,
        config: Optional[dict] = None) -> RAGAnswer:
    """
    Main RAG entry point.
    config keys: top_k_vector, top_k_graph, rerank (bool), explain (bool)
    """
    if config is None:
        config = {}

    top_k_vec   = config.get("top_k_vector", 10)
    top_k_graph = config.get("top_k_graph", 10)
    do_rerank   = config.get("rerank", True)
    explain     = config.get("explain", True)

    pipeline_steps = []

    # Step 1: Vector retrieval
    pipeline_steps.append("Vector Retrieval")
    vec_results = vector_search(question, company, top_k=top_k_vec)

    # Step 2: Graph retrieval
    pipeline_steps.append("Graph Retrieval")
    # Extract topic keywords from question
    topic = question.lower().replace("what is the", "").replace("what are the", "").strip()
    graph_results = graph_search(company, topic=topic, top_k=top_k_graph)

    # Step 3: Combine & Rerank
    if do_rerank and vec_results:
        pipeline_steps.append("Reranker")
        combined = vec_results  # Graph results don't have text suitable for reranking
        vec_results = rerank(question, combined, top_k=top_k_vec)
    pipeline_steps.append("KG-RAG Reasoner")

    # Step 4: Generate answer
    answer_text = generate_answer(question, company, vec_results, graph_results)

    # Build evidence spans
    evidence_spans = []
    for chunk in vec_results[:5]:
        evidence_spans.append(EvidenceSpan(
            doc_type="10-K",
            section=chunk.get("section", "Item 1"),
            page=chunk.get("page_estimate", 0),
            confidence=round(chunk.get("score", 0), 2),
            text=chunk.get("text", "")[:300],
            chunk_id=chunk.get("chunk_id", ""),
        ))

    # Build KG paths
    kg_paths = []
    if explain:
        kg_paths = get_kg_paths(company)

    return RAGAnswer(
        question=question,
        company=company,
        answer=answer_text,
        evidence_spans=evidence_spans,
        kg_paths=kg_paths,
        pipeline_steps=pipeline_steps,
    )


def get_company_overview(company: str) -> dict:
    """Get overview data for a company from KG."""
    G = load_kg()
    company_node = f"Company:{company}"

    overview = {
        "ticker": company,
        "name": "",
        "mission": "",
        "objectives": [],
        "capabilities": [],
        "initiatives": [],
        "risks": [],
        "kg_stats": {"nodes": 0, "edges": 0},
    }

    if company_node not in G:
        return overview

    node_data = G.nodes[company_node]
    overview["name"] = node_data.get("name", company)

    # Count company-specific nodes
    company_nodes = [company_node]
    company_edges = 0

    for neighbor in G.neighbors(company_node):
        label = G.nodes[neighbor].get(":LABEL", "")
        text = G.nodes[neighbor].get("text", "") or G.nodes[neighbor].get("name", "")
        relation = G[company_node][neighbor].get("relation", "")
        company_nodes.append(neighbor)
        company_edges += 1

        if label == "Mission":
            overview["mission"] = text
        elif label == "StrategicObjective":
            overview["objectives"].append(text)
        elif label == "Capability":
            overview["capabilities"].append(text)
        elif label == "Initiative":
            overview["initiatives"].append(text)
        elif label == "RiskTheme":
            overview["risks"].append(text)

        # 2nd hop
        for n2 in G.neighbors(neighbor):
            if n2 != company_node:
                company_nodes.append(n2)
                company_edges += 1

    overview["kg_stats"] = {
        "nodes": len(set(company_nodes)),
        "edges": company_edges,
    }

    return overview


def compare_companies(company1: str, company2: str) -> dict:
    """Compare two companies across KG dimensions."""
    o1 = get_company_overview(company1)
    o2 = get_company_overview(company2)

    return {
        "company1": o1,
        "company2": o2,
        "dimensions": ["Mission", "Strategic Objectives", "Capabilities",
                       "Initiatives", "Risk Themes"],
    }
