"""
RAG Engine: Hybrid Vector + Graph Retrieval (v2)
=================================================
Core retrieval logic for the MSG-KG system. Implements hybrid Graph-RAG
combining semantic vector search with structured KG traversal.

Pipeline:
1. Vector Retrieval: FAISS + all-MiniLM-L6-v2 on 10-K chunks
2. Graph Retrieval: TTL-parsed KG with entity linking + subgraph extraction
3. Reranking: Cross-encoder re-scoring of combined evidence
4. Context Fusion: Merge vector + graph context for LLM prompt
5. LLM Synthesis: Mixtral-8x7B via HuggingFace Inference API
"""

import os, json, pathlib, re
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = pathlib.Path(__file__).parent
CHUNKS_DIR = BASE_DIR / "data" / "chunks"
KG_DIR     = BASE_DIR / "MSGKG"

# ── Lazy-loaded globals ──────────────────────────────────────────────────
_embedder = None
_reranker = None
_faiss_index = None
_all_chunks = None
_kg_graph = None


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class EvidenceSpan:
    doc_type: str = "10-K"
    section: str = "Item 1"
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


# ── Company Metadata (updated for new companies) ────────────────────────
COMPANY_META = {
    "AMD": {
        "name": "Advanced Micro Devices, Inc.",
        "mission": "Build great products that accelerate next-generation computing experiences.",
        "sector": "Semiconductors",
        "fiscal_year": "FY2024",
    },
    "ALX": {
        "name": "Alexander's, Inc.",
        "mission": "Real estate leasing operations primarily in the New York metropolitan area.",
        "sector": "Real Estate (REIT)",
        "fiscal_year": "FY2024",
    },
    "LNG": {
        "name": "Cheniere Energy, Inc.",
        "mission": "To be the world's leading full-service LNG provider.",
        "sector": "Energy (LNG)",
        "fiscal_year": "FY2024",
    },
    "UNM": {
        "name": "Unum Group",
        "mission": "Helping the working world thrive throughout life's moments.",
        "sector": "Insurance",
        "fiscal_year": "FY2024",
    },
    "AAL": {
        "name": "American Airlines Group Inc.",
        "mission": "To care for people on life's journey.",
        "sector": "Airlines",
        "fiscal_year": "FY2024",
    },
    "WRB": {
        "name": "W. R. Berkley Corporation",
        "mission": "Customer-focused insurance execution with decentralized underwriting.",
        "sector": "Insurance",
        "fiscal_year": "FY2024",
    },
}


# ── Embedding Model ─────────────────────────────────────────────────────

def get_embedder():
    """Lazy-load retrieval-optimized embedding model (all-MiniLM-L6-v2)."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model (all-MiniLM-L6-v2)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[OK] Embedding model loaded")
        except Exception as e:
            print(f"[ERR] Could not load embedding model: {e}")
            raise
    return _embedder


def get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _reranker = False  # sentinel: tried and failed
    return _reranker if _reranker is not False else None


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
                data = json.load(f)
                # Only load chunks for companies we care about
                if data and isinstance(data, list):
                    chunks.extend(data)
    _all_chunks = chunks
    print(f"[OK] Loaded {len(chunks)} chunks from {CHUNKS_DIR}")
    return chunks


def build_faiss_index(chunks: list[dict] = None):
    """Build FAISS index from chunk embeddings."""
    global _faiss_index, _all_chunks
    import faiss

    if chunks is None:
        chunks = load_all_chunks()

    if not chunks:
        print("[WARN] No chunks to index")
        return None

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _faiss_index = index
    _all_chunks = chunks
    print(f"[OK] FAISS index: {index.ntotal} vectors, dim={dim}")
    return index


def vector_search(query: str, company: str = None, top_k: int = 10) -> list[dict]:
    """Semantic search using FAISS index."""
    global _faiss_index, _all_chunks

    if _faiss_index is None:
        build_faiss_index()

    if _faiss_index is None or not _all_chunks:
        return []

    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    import faiss
    faiss.normalize_L2(q_emb)

    search_k = min(top_k * 5, _faiss_index.ntotal)
    scores, indices = _faiss_index.search(q_emb, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = _all_chunks[idx].copy()
        chunk["score"] = float(score)
        chunk["source"] = "vector"

        if company and chunk.get("ticker") != company:
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results


# ── TTL Parser for Knowledge Graph (rdflib-based) ───────────────────────

def _local_name(uri) -> str:
    """Extract the local name from an RDF URI. e.g. http://msgkg.io/instances#Company_AMD -> Company_AMD"""
    s = str(uri)
    for sep in ('#', '/'):
        if sep in s:
            return s.rsplit(sep, 1)[-1]
    return s


def _build_kg_from_ttl(G: nx.DiGraph, filepath: pathlib.Path, source: str = ""):
    """
    Parse a Turtle (.ttl) file using rdflib and add triples to the NetworkX graph.
    
    - rdf:type triples set the :LABEL attribute on the subject node
    - Literal-valued triples become node attributes
    - URI-valued triples become directed edges
    """
    from rdflib import Graph as RDFGraph, RDF, Literal

    rdf_graph = RDFGraph()
    rdf_graph.parse(str(filepath), format="turtle")

    for subj, pred, obj in rdf_graph:
        subj_name = _local_name(subj)
        pred_name = _local_name(pred)

        # Ensure subject node exists
        if subj_name not in G:
            G.add_node(subj_name, source=source)

        if pred == RDF.type:
            # rdf:type -> set the node's class label
            obj_name = _local_name(obj)
            G.nodes[subj_name][":LABEL"] = obj_name
        elif isinstance(obj, Literal):
            # Data property: store as node attribute
            val = str(obj)
            G.nodes[subj_name][pred_name] = val
            # Use descriptive predicates as display text
            if pred_name in ("companyName", "description", "label", "comment",
                            "riskDescription", "initiativeDescription",
                            "objectiveDescription", "metricDescription",
                            "metricName", "riskName", "initiativeName"):
                G.nodes[subj_name]["text"] = val
                G.nodes[subj_name]["name"] = val
        else:
            # Object property: add directed edge
            obj_name = _local_name(obj)
            if obj_name not in G:
                G.add_node(obj_name, source=source)
            G.add_edge(subj_name, obj_name, relation=pred_name, source=source)


# ── Knowledge Graph Loading ─────────────────────────────────────────────

def load_kg() -> nx.DiGraph:
    """
    Load KG from TTL files into NetworkX DiGraph.
    Loads: kg1.ttl (ontology instances), AMD.ttl, ALX.ttl, LNG.ttl (company data).
    Falls back to legacy nodes.csv/edges.csv if TTL files are missing.
    """
    global _kg_graph
    if _kg_graph is not None:
        return _kg_graph

    G = nx.DiGraph()
    loaded_ttl = False

    # Primary: Load from TTL files
    ttl_files = [
        ("kg1.ttl", "kg1"),
        ("AMD.ttl", "AMD"),
        ("ALX.ttl", "ALX"),
        ("LNG.ttl", "LNG"),
    ]

    for fname, source in ttl_files:
        fpath = KG_DIR / fname
        if fpath.exists():
            try:
                _build_kg_from_ttl(G, fpath, source=source)
                loaded_ttl = True
            except Exception as e:
                print(f"[WARN] Failed to parse {fname}: {e}")

    # Fallback: Load from CSV if no TTL files worked
    if not loaded_ttl:
        import pandas as pd
        nodes_path = KG_DIR / "nodes.csv"
        edges_path = KG_DIR / "edges.csv"

        if nodes_path.exists():
            df_nodes = pd.read_csv(nodes_path, encoding="utf-8")
            for _, row in df_nodes.iterrows():
                node_id = row[":ID"]
                attrs = {k: v for k, v in row.items()
                         if k != ":ID" and not (isinstance(v, float) and np.isnan(v)) and str(v).strip()}
                G.add_node(node_id, **attrs)

        if edges_path.exists():
            df_edges = pd.read_csv(edges_path, encoding="utf-8")
            for _, row in df_edges.iterrows():
                G.add_edge(row[":START_ID"], row[":END_ID"],
                           relation=row[":TYPE"])

    _kg_graph = G
    print(f"[OK] KG loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Graph RAG: Entity Linking + Subgraph Extraction ────────────────────

def _extract_query_entities(query: str) -> list[str]:
    """Extract potential KG entity mentions from a user query."""
    # Normalize
    q = query.lower().strip()

    # Known concept keywords that map to KG node types
    concept_keywords = {
        "mission": ["Mission", "mission", "hasMission"],
        "strategy": ["StrategicObjective", "Strategy", "hasStrategicObjective"],
        "risk": ["Risk", "RiskTheme", "hasRiskTheme", "hasRisk"],
        "initiative": ["Initiative", "hasInitiative"],
        "capability": ["Capability", "hasCapability"],
        "human capital": ["HumanCapitalMetric", "HumanCapitalStrategy",
                         "hasHumanCapitalStrategy", "hasHumanCapitalMetric"],
        "hcm": ["HumanCapitalMetric", "HumanCapitalStrategy"],
        "employee": ["HumanCapitalMetric", "hasHumanCapitalMetric"],
        "compensation": ["CompensationPhilosophy", "hasCompensationPhilosophy"],
        "culture": ["promotesCulture", "Culture"],
        "financial": ["FinancialMetric", "hasFinancialMetric"],
        "revenue": ["FinancialMetric", "Revenue"],
        "diversity": ["supportsGroup", "DiversityGroup"],
        "engagement": ["usesEngagementTool", "EngagementTool"],
        "product": ["Product", "hasProduct"],
        "competition": ["Competition", "Competitor"],
        "ai": ["AI", "ArtificialIntelligence"],
        "data center": ["DataCenter"],
        "lng": ["LNG", "LiquefiedNaturalGas"],
        "semiconductor": ["Semiconductor"],
        "real estate": ["RealEstate"],
    }

    entities = []
    for keyword, kg_types in concept_keywords.items():
        if keyword in q:
            entities.extend(kg_types)

    return list(set(entities))


def _find_company_nodes(G: nx.DiGraph, company: str) -> list[str]:
    """Find all KG nodes representing a company, sorted by connectivity (most edges first)."""
    candidates = [
        f"Company_{company}",          # TTL: inst:Company_AMD -> Company_AMD
        f"Company:{company}",           # Legacy CSV format
        company.upper(),                # Direct ticker
        company,                        # As-is
    ]
    
    found = []
    for node in G.nodes():
        node_str = str(node)
        if node_str in candidates:
            found.append(node_str)
        elif node_str.endswith(f"_{company}") and G.nodes[node].get(":LABEL") == "Company":
            found.append(node_str)
    
    # Sort by out_degree descending so richest node (from company TTL) comes first
    found.sort(key=lambda n: G.out_degree(n), reverse=True)
    return found


def graph_search(company: str, topic: str = None, top_k: int = 10) -> list[dict]:
    """
    Enhanced Graph RAG: Entity linking + subgraph extraction.
    1. Link query entities to KG nodes
    2. Extract relevant subgraph around company + matched entities
    3. Score paths by relevance
    """
    G = load_kg()

    # Find company root node(s)
    company_nodes = _find_company_nodes(G, company)

    if not company_nodes:
        return []

    # Extract query entities
    query_entities = _extract_query_entities(topic or "")

    results = []

    for company_node in company_nodes:
        # 1-hop neighbors
        for neighbor in G.neighbors(company_node):
            edge_data = G[company_node][neighbor]
            node_data = G.nodes[neighbor]
            label = node_data.get(":LABEL", "Unknown")
            text = (node_data.get("text", "") or node_data.get("name", "") or
                    node_data.get("label", "") or str(neighbor))
            relation = edge_data.get("relation", "")

            # Score based on entity matching
            score = 0.5
            if query_entities:
                for qe in query_entities:
                    qe_lower = qe.lower()
                    if (qe_lower in label.lower() or
                        qe_lower in relation.lower() or
                        qe_lower in text.lower()):
                        score = 0.9
                        break
                    elif topic and any(w in text.lower() for w in topic.lower().split()[:3]):
                        score = max(score, 0.7)

            results.append({
                "node_id": neighbor,
                "label": label,
                "text": text[:300],
                "relation": relation,
                "score": score,
                "hop": 1,
                "source": "kg",
            })

            # 2-hop neighbors
            for n2 in G.neighbors(neighbor):
                if n2 == company_node:
                    continue
                n2_data = G.nodes[n2]
                e2 = G[neighbor][n2].get("relation", "?")
                n2_label = n2_data.get(":LABEL", "Unknown")
                n2_text = (n2_data.get("text", "") or n2_data.get("name", "") or
                          n2_data.get("label", "") or str(n2))

                hop2_score = 0.3
                if query_entities:
                    for qe in query_entities:
                        if qe.lower() in n2_label.lower() or qe.lower() in n2_text.lower():
                            hop2_score = 0.7
                            break

                results.append({
                    "node_id": n2,
                    "label": n2_label,
                    "text": n2_text[:300],
                    "relation": f"{relation} -> {e2}",
                    "score": hop2_score,
                    "hop": 2,
                    "path": f"{company_node} -[{relation}]-> {label} -[{e2}]-> {n2_label}",
                    "source": "kg",
                })

    # Deduplicate by node_id, keeping highest score
    seen = {}
    for r in results:
        nid = r["node_id"]
        if nid not in seen or r["score"] > seen[nid]["score"]:
            seen[nid] = r
    results = list(seen.values())

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def get_kg_paths(company: str) -> list[KGPath]:
    """Extract meaningful KG paths for explain mode."""
    G = load_kg()

    # Find company root
    company_nodes = _find_company_nodes(G, company)
    if not company_nodes:
        return []
    company_node = company_nodes[0]

    paths = []
    for n1 in G.neighbors(company_node):
        n1_label = G.nodes[n1].get(":LABEL", "")
        e1 = G[company_node][n1].get("relation", "")
        n1_text = G.nodes[n1].get("text", "") or G.nodes[n1].get("name", "") or str(n1)

        paths.append(KGPath(
            path=f"{company} -[{e1}]-> {n1_label}",
            description=n1_text[:200],
        ))

        for n2 in G.neighbors(n1):
            if n2 != company_node:
                n2_label = G.nodes[n2].get(":LABEL", "")
                e2 = G[n1][n2].get("relation", "")
                n2_text = G.nodes[n2].get("text", "") or G.nodes[n2].get("name", "") or str(n2)

                paths.append(KGPath(
                    path=f"{company} -> {n1_label} -[{e2}]-> {n2_label}",
                    description=f"{n1_text[:80]}... -> {n2_text[:80]}",
                ))

    return paths


# ── Reranker ────────────────────────────────────────────────────────────

def rerank(query: str, results: list[dict], top_k: int = 10) -> list[dict]:
    """Cross-encoder reranking. Falls back to score-based ranking."""
    reranker = get_reranker()
    if reranker:
        try:
            pairs = [(query, r.get("text", "")) for r in results]
            scores = reranker.predict(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception:
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
    else:
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return results[:top_k]


# ── Context Fusion ──────────────────────────────────────────────────────

def _fuse_context(vec_results: list[dict], kg_results: list[dict],
                  max_tokens: int = 3000) -> tuple[str, str]:
    """
    Fuse vector search results with KG subgraph into coherent context.
    Returns (text_context, kg_context) formatted for the prompt.
    """
    # Text context from vector search
    context_parts = []
    word_budget = max_tokens
    for i, chunk in enumerate(vec_results[:8]):
        score = chunk.get("rerank_score", chunk.get("score", 0))
        section = chunk.get("section", "Item 1")
        page = chunk.get("page_estimate", "?")
        text = chunk["text"][:600]
        words_used = len(text.split())

        if word_budget - words_used < 0:
            break

        context_parts.append(
            f"[Evidence {i+1}] (10-K {section}, Page ~{page}, Score: {score:.2f})\n{text}"
        )
        word_budget -= words_used

    # KG context — structured by relationship type
    kg_by_type = {}
    for item in kg_results[:15]:
        label = item.get("label", "Other")
        text = item.get("text", "")
        relation = item.get("relation", "")
        hop = item.get("hop", 1)
        if not text or text == label:
            continue
        if label not in kg_by_type:
            kg_by_type[label] = []
        entry = f"{text[:200]}"
        if relation:
            entry += f" (via {relation})"
        if hop > 1:
            entry += f" [{hop}-hop]"
        kg_by_type[label].append(entry)

    kg_parts = []
    for label, items in kg_by_type.items():
        kg_parts.append(f"  [{label}]")
        for it in items[:3]:
            kg_parts.append(f"    - {it}")

    text_context = "\n\n".join(context_parts) if context_parts else "No 10-K evidence retrieved."
    kg_context = "\n".join(kg_parts) if kg_parts else "No KG context available."

    return text_context, kg_context


# ── LLM Generation ──────────────────────────────────────────────────────

def generate_answer(question: str, company: str, context_chunks: list[dict],
                    kg_context: list[dict]) -> str:
    """Generate answer using HuggingFace Inference API (Mixtral-8x7B)."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    meta = COMPANY_META.get(company, {})
    company_name = meta.get("name", company)
    mission = meta.get("mission", "N/A")
    sector = meta.get("sector", "N/A")

    # Fuse vector + KG context
    text_context, kg_text = _fuse_context(context_chunks, kg_context)

    prompt = f"""<s>[INST] You are a financial analyst specializing in the MSG-KG (Mission-Strategy-Goals Knowledge Graph) framework. Analyze SEC 10-K filings using both document evidence and knowledge graph data.

COMPANY PROFILE:
- Name: {company_name} ({company})
- Mission: {mission}
- Sector: {sector}

10-K FILING EVIDENCE:
{text_context}

KNOWLEDGE GRAPH DATA:
{kg_text}

QUESTION: {question}

Instructions:
1. Ground your answer in the retrieved 10-K evidence -- cite specific passages
2. Reference knowledge graph relationships where they strengthen your analysis
3. Connect findings to the company's mission and strategy
4. Evaluate Human Capital Management disclosures if relevant
5. If evidence is insufficient, clearly state what is missing
6. Be concise but thorough -- aim for 3-5 key insights

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
            # Clean up trailing incomplete sentences
            if answer and answer[-1] not in '.!?"':
                last_period = answer.rfind('.')
                if last_period > len(answer) * 0.5:
                    answer = answer[:last_period + 1]
            return answer
        except Exception as e:
            print(f"[WARN] LLM generation failed: {e}")

    # Fallback: structured extractive summary
    parts = []
    parts.append(f"**{company_name}** ({sector})")
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

    return "\n\n".join(parts) if parts else f"No relevant information found for {company}."


# ── Main RAG Pipeline ───────────────────────────────────────────────────

def ask(question: str, company: str,
        config: Optional[dict] = None) -> RAGAnswer:
    """
    Main RAG entry point.
    Config keys: top_k_vector, top_k_graph, rerank (bool), explain (bool)
    """
    if config is None:
        config = {}

    top_k_vec   = config.get("top_k_vector", 10)
    top_k_graph = config.get("top_k_graph", 15)
    do_rerank   = config.get("rerank", True)
    explain     = config.get("explain", True)

    pipeline_steps = []

    # Step 1: Vector retrieval
    pipeline_steps.append("Vector Retrieval (MiniLM-L6)")
    vec_results = vector_search(question, company, top_k=top_k_vec)

    # Step 2: Graph RAG — entity-linked subgraph extraction
    pipeline_steps.append("Graph RAG (Entity Linking + Subgraph)")
    topic = question.lower()
    # Remove common question prefixes
    for prefix in ["what is the", "what are the", "how does", "tell me about", "describe"]:
        topic = topic.replace(prefix, "")
    topic = topic.strip()
    graph_results = graph_search(company, topic=topic, top_k=top_k_graph)

    # Step 3: Rerank vector results
    if do_rerank and vec_results:
        pipeline_steps.append("Cross-Encoder Reranking")
        vec_results = rerank(question, vec_results, top_k=top_k_vec)

    # Step 4: Context Fusion + LLM Generation
    pipeline_steps.append("Context Fusion + LLM")
    answer_text = generate_answer(question, company, vec_results, graph_results)

    # Build evidence spans
    evidence_spans = []
    for chunk in vec_results[:5]:
        evidence_spans.append(EvidenceSpan(
            doc_type="10-K",
            section=chunk.get("section", "Item 1"),
            page=chunk.get("page_estimate", 0),
            confidence=round(chunk.get("rerank_score", chunk.get("score", 0)), 2),
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


# ── Company Overview (for app.py) ───────────────────────────────────────

def get_company_overview(company: str) -> dict:
    """Get overview data for a company from KG by merging ALL company nodes."""
    G = load_kg()

    overview = {
        "ticker": company,
        "name": COMPANY_META.get(company, {}).get("name", company),
        "mission": COMPANY_META.get(company, {}).get("mission", ""),
        "objectives": [],
        "capabilities": [],
        "initiatives": [],
        "risks": [],
        "hcm_metrics": [],
        "financial_metrics": [],
        "kg_stats": {"nodes": 0, "edges": 0},
    }

    # Find ALL company root nodes (may be multiple: AMD from kg1.ttl, Company_AMD from AMD.ttl)
    root_nodes = _find_company_nodes(G, company)
    if not root_nodes:
        return overview

    # Use the most-connected node for the name
    primary_node = root_nodes[0]  # already sorted by out_degree
    node_data = G.nodes[primary_node]
    if node_data.get("text"):
        overview["name"] = node_data["text"]

    seen_nodes = set(root_nodes)
    total_edges = 0
    seen_texts = set()  # deduplicate

    # Traverse neighbors of ALL company nodes to merge data
    for company_node in root_nodes:
        for neighbor in G.neighbors(company_node):
            label = G.nodes[neighbor].get(":LABEL", "")
            text = (G.nodes[neighbor].get("text", "") or
                    G.nodes[neighbor].get("name", "") or str(neighbor))
            seen_nodes.add(neighbor)
            total_edges += 1

            # Deduplicate by text
            if text in seen_texts:
                continue
            seen_texts.add(text)

            if "Mission" in label:
                overview["mission"] = text
            elif "StrategicObjective" in label or "Strategy" in label:
                overview["objectives"].append(text)
            elif "Capability" in label:
                overview["capabilities"].append(text)
            elif "Initiative" in label:
                overview["initiatives"].append(text)
            elif "Risk" in label:
                overview["risks"].append(text)
            elif "HumanCapital" in label:
                overview["hcm_metrics"].append(text)
            elif "Financial" in label:
                overview["financial_metrics"].append(text)

            # 2nd hop
            for n2 in G.neighbors(neighbor):
                if n2 not in root_nodes:
                    seen_nodes.add(n2)
                    total_edges += 1

    overview["kg_stats"] = {
        "nodes": len(seen_nodes),
        "edges": total_edges,
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
                       "Initiatives", "Risk Themes", "HCM Metrics"],
    }
