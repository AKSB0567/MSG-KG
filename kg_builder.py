"""
KG Builder: Strategic Extraction Pipeline
=========================================
Constructs the Mission-Strategy-Goals Knowledge Graph triples for peer companies.
Uses a hybrid extraction strategy:
- LLM Extraction: Mixtral/Mistral via HuggingFace Inference API for high-fidelity concepts.
- Regex Fallback: Pattern-based extraction to ensure baseline data availability.
"""

import os, re, json, pathlib, csv
from dotenv import load_dotenv

load_dotenv()

BASE_DIR    = pathlib.Path(__file__).parent
ITEM1_DIR   = BASE_DIR / "data" / "item1"
KG_DIR      = BASE_DIR / "MSGKG"
KG_JSON_DIR = BASE_DIR / "data" / "kg_json"

TICKERS = ["AAPL", "AMD", "TGT", "WMT", "TSN", "MSFT"]

COMPANY_META = {
    "AAPL": {"cik": "320193",  "name": "Apple Inc.",               "fiscal_year": "FY2024"},
    "AMD":  {"cik": "2488",    "name": "Advanced Micro Devices",   "fiscal_year": "FY2024"},
    "TGT":  {"cik": "27419",   "name": "Target Corporation",       "fiscal_year": "FY2025"},
    "WMT":  {"cik": "104169",  "name": "Walmart Inc.",             "fiscal_year": "FY2025"},
    "TSN":  {"cik": "100493",  "name": "Tyson Foods Inc.",         "fiscal_year": "FY2025"},
    "MSFT": {"cik": "789019",  "name": "Microsoft Corporation",    "fiscal_year": "FY2025"},
}

KG_EXTRACTION_PROMPT = """You are a financial knowledge graph extraction assistant.

Given the following Item 1 (Business) text from a 10-K filing for {company_name} (ticker: {ticker}), extract structured knowledge graph elements.

Return a JSON object with these fields:
- "Mission": {{"text": "..."}} — the company's stated mission or purpose
- "Vision": {{"text": "..."}} — forward-looking aspirations (if found)
- "StrategicObjective": [{{"text": "...", "objective_type": "growth|cost_leadership|innovation|diversification|sustainability"}}]
- "Capability": [{{"name": "..."}}] — key organizational/technical capabilities
- "Initiative": [{{"name": "...", "status": "active|expanding|planned|completed"}}] — specific programs/projects
- "RiskTheme": [{{"name": "...", "risk_category": "macro|regulatory|operational|competitive|technology|supply_chain"}}]

Rules:
- Extract 2-4 items per category
- Extract complete, detailed sentences (max 100 words per item)
- Only extract what is explicitly stated in the text
- Return valid JSON only, no markdown

TEXT:
{text_excerpt}

JSON:"""


def extract_kg_with_llm(ticker: str, text: str) -> dict | None:
    """Use HuggingFace Inference API to extract KG triples."""
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("  ⚠ No HUGGINGFACE_TOKEN found, skipping LLM extraction")
        return None

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=hf_token)
        meta = COMPANY_META[ticker]

        # Use first ~3000 words to stay in context limits
        words = text.split()
        excerpt = " ".join(words[:3000])

        prompt = KG_EXTRACTION_PROMPT.format(
            company_name=meta["name"],
            ticker=ticker,
            text_excerpt=excerpt,
        )

        response = client.text_generation(
            prompt,
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            max_new_tokens=1024,
            temperature=0.1,
            return_full_text=False,
        )

        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"  ⚠ LLM extraction failed for {ticker}: {e}")

    return None


def extract_kg_rulebased(ticker: str, text: str) -> dict:
    """
    Rule-based fallback KG extraction using regex patterns.
    Extracts mission, objectives, capabilities, risks from Item 1 text.
    """
    result = {}
    text_lower = text.lower()

    # ── Mission extraction ───────────────────────────────────────
    mission_patterns = [
        r'(?:our\s+)?mission\s+(?:is\s+)?(?:to\s+)?([^.]+\.)',
        r'(?:our\s+)?purpose\s+(?:is\s+)?(?:to\s+)?([^.]+\.)',
        r'we\s+(?:are\s+)?(?:committed|dedicated)\s+to\s+([^.]+\.)',
        r'(?:our\s+)?goal\s+is\s+to\s+([^.]+\.)',
    ]
    for pat in mission_patterns:
        m = re.search(pat, text_lower)
        if m:
            result["Mission"] = {"text": m.group(1).strip()[:500]}
            break
    if "Mission" not in result:
        # Take first meaningful sentence as proxy
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 40]
        if sentences:
            result["Mission"] = {"text": sentences[0][:500]}

    # ── Strategic Objectives ─────────────────────────────────────
    obj_keywords = {
        "growth": ["expand", "grow", "increase market", "scale", "enter new market"],
        "innovation": ["innovat", "research and development", "r&d", "technology leadership",
                       "artificial intelligence", "machine learning"],
        "cost_leadership": ["cost efficien", "low cost", "price leadership", "operational efficien"],
        "diversification": ["diversif", "new segment", "portfolio expansion"],
        "sustainability": ["sustainab", "environmental", "carbon", "renewable", "ESG"],
    }
    objectives = []
    for obj_type, keywords in obj_keywords.items():
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx >= 0:
                # Extract surrounding sentence
                start = max(0, text_lower.rfind('.', 0, idx) + 1)
                end = text_lower.find('.', idx)
                if end < 0:
                    end = min(len(text_lower), idx + 200)
                sentence = text[start:end+1].strip()
                if len(sentence) > 20:
                    objectives.append({
                        "text": sentence[:500],
                        "objective_type": obj_type,
                    })
                    break
    result["StrategicObjective"] = objectives[:4]

    # ── Capabilities ─────────────────────────────────────────────
    cap_keywords = ["supply chain", "logistics", "digital platform", "cloud",
                    "artificial intelligence", "data analytics", "manufacturing",
                    "distribution network", "e-commerce", "technology infrastructure",
                    "research and development", "brand management"]
    capabilities = []
    for kw in cap_keywords:
        if kw in text_lower:
            capabilities.append({"name": kw})
    result["Capability"] = capabilities[:5]

    # ── Initiatives ──────────────────────────────────────────────
    init_patterns = [
        r'(?:launched|introduced|implemented|expanded)\s+(?:our\s+|the\s+)?([A-Z][^,.]+)',
        r'(?:our\s+)?([A-Z][A-Za-z+\s]+(?:platform|program|initiative|service|membership))',
    ]
    initiatives = []
    for pat in init_patterns:
        for m in re.finditer(pat, text):
            name = m.group(1).strip()
            if 10 < len(name) < 80:
                initiatives.append({"name": name, "status": "active"})
                if len(initiatives) >= 3:
                    break
        if len(initiatives) >= 3:
            break
    result["Initiative"] = initiatives[:3]

    # ── Risk Themes ──────────────────────────────────────────────
    risk_map = {
        "macro": ["inflation", "economic", "recession", "interest rate", "currency"],
        "regulatory": ["regulat", "compliance", "government", "legal", "legislation"],
        "competitive": ["competi", "market share", "pricing pressure"],
        "supply_chain": ["supply chain", "logistics", "inventory", "supplier"],
        "technology": ["cybersecurity", "cyber", "data breach", "technology risk"],
        "operational": ["workforce", "labor", "operational risk", "disruption"],
    }
    risks = []
    for cat, keywords in risk_map.items():
        for kw in keywords:
            if kw in text_lower:
                risks.append({"name": kw, "risk_category": cat})
                break
    result["RiskTheme"] = risks[:5]

    return result


def build_kg_for_company(ticker: str) -> dict | None:
    """Extract KG for one company: try LLM first, fallback to rules."""
    txt_path = ITEM1_DIR / f"{ticker}_item1.txt"
    if not txt_path.exists():
        print(f"  ✗ No Item 1 text found for {ticker}")
        return None

    text = txt_path.read_text(encoding="utf-8")
    if len(text) < 100:
        print(f"  ✗ Item 1 text too short for {ticker}")
        return None

    print(f"  Attempting LLM extraction for {ticker}...")
    kg = extract_kg_with_llm(ticker, text)

    if not kg:
        print(f"  Using rule-based extraction for {ticker}...")
        kg = extract_kg_rulebased(ticker, text)

    # Add company metadata
    meta = COMPANY_META[ticker]
    kg["Company"] = {"ticker": ticker, "cik": meta["cik"]}

    # Build edges
    edges = []
    if "Mission" in kg:
        edges.append(["Company", "hasMission", "Mission"])
    for obj in kg.get("StrategicObjective", []):
        edges.append(["Company", "statesObjective", obj.get("text", "")[:50]])
    for cap in kg.get("Capability", []):
        edges.append(["Company", "buildsCapability", cap.get("name", "")])
        # Link capabilities to first objective if exists
        if kg.get("StrategicObjective"):
            edges.append(["Capability", "supportsObjective",
                          kg["StrategicObjective"][0].get("text", "")[:50]])
    for init in kg.get("Initiative", []):
        edges.append(["Company", "pursues", init.get("name", "")])
        if kg.get("Capability"):
            edges.append(["Initiative", "initiativeEnables",
                          kg["Capability"][0].get("name", "")])
    for risk in kg.get("RiskTheme", []):
        edges.append(["Company", "exposedTo", risk.get("name", "")])

    kg["Edges"] = edges
    return kg


def slugify(text: str) -> str:
    """Convert text to URL-safe slug."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')[:60]


def merge_to_csv(all_kg: dict[str, dict]):
    """Merge all per-company KG JSONs into unified nodes.csv and edges.csv."""
    node_rows = []
    edge_rows = []

    for ticker, kg in all_kg.items():
        meta = COMPANY_META[ticker]

        # Company node
        node_rows.append({
            ":ID": f"Company:{ticker}",
            ":LABEL": "Company",
            "ticker": ticker,
            "cik": meta["cik"],
            "name": meta["name"],
            "text": "",
            "objective_type": "",
            "status": "",
            "risk_category": "",
        })

        # Mission
        if "Mission" in kg and kg["Mission"]:
            mission_text = kg["Mission"].get("text", "")
            node_rows.append({
                ":ID": f"Mission:{ticker}:mission",
                ":LABEL": "Mission",
                "ticker": "", "cik": "", "name": "",
                "text": mission_text,
                "objective_type": "", "status": "", "risk_category": "",
            })
            edge_rows.append({
                ":START_ID": f"Company:{ticker}",
                ":END_ID": f"Mission:{ticker}:mission",
                ":TYPE": "hasMission",
            })

        # Strategic Objectives
        for obj in kg.get("StrategicObjective", []):
            slug = slugify(obj.get("text", "objective"))
            node_id = f"StrategicObjective:{ticker}:{slug}"
            node_rows.append({
                ":ID": node_id, ":LABEL": "StrategicObjective",
                "ticker": "", "cik": "", "name": "",
                "text": obj.get("text", ""),
                "objective_type": obj.get("objective_type", ""),
                "status": "", "risk_category": "",
            })
            edge_rows.append({
                ":START_ID": f"Company:{ticker}",
                ":END_ID": node_id,
                ":TYPE": "statesObjective",
            })

        # Capabilities
        for cap in kg.get("Capability", []):
            cap_name = cap.get("name", "capability")
            slug = slugify(cap_name)
            node_id = f"Capability:{ticker}:{slug}"
            node_rows.append({
                ":ID": node_id, ":LABEL": "Capability",
                "ticker": "", "cik": "", "name": cap_name,
                "text": "", "objective_type": "", "status": "", "risk_category": "",
            })
            edge_rows.append({
                ":START_ID": f"Company:{ticker}",
                ":END_ID": node_id,
                ":TYPE": "buildsCapability",
            })
            # Link to first objective
            objs = kg.get("StrategicObjective", [])
            if objs:
                obj_slug = slugify(objs[0].get("text", ""))
                edge_rows.append({
                    ":START_ID": node_id,
                    ":END_ID": f"StrategicObjective:{ticker}:{obj_slug}",
                    ":TYPE": "supportsObjective",
                })

        # Initiatives
        for init in kg.get("Initiative", []):
            init_name = init.get("name", "initiative")
            slug = slugify(init_name)
            node_id = f"Initiative:{ticker}:{slug}"
            node_rows.append({
                ":ID": node_id, ":LABEL": "Initiative",
                "ticker": "", "cik": "", "name": init_name,
                "text": "", "objective_type": "",
                "status": init.get("status", "active"),
                "risk_category": "",
            })
            edge_rows.append({
                ":START_ID": f"Company:{ticker}",
                ":END_ID": node_id,
                ":TYPE": "pursues",
            })
            caps = kg.get("Capability", [])
            if caps:
                cap_slug = slugify(caps[0].get("name", ""))
                edge_rows.append({
                    ":START_ID": node_id,
                    ":END_ID": f"Capability:{ticker}:{cap_slug}",
                    ":TYPE": "initiativeEnables",
                })

        # Risk Themes
        for risk in kg.get("RiskTheme", []):
            risk_name = risk.get("name", "risk")
            slug = slugify(risk_name)
            node_id = f"RiskTheme:{ticker}:{slug}"
            node_rows.append({
                ":ID": node_id, ":LABEL": "RiskTheme",
                "ticker": "", "cik": "", "name": risk_name,
                "text": "", "objective_type": "", "status": "",
                "risk_category": risk.get("risk_category", ""),
            })
            edge_rows.append({
                ":START_ID": f"Company:{ticker}",
                ":END_ID": node_id,
                ":TYPE": "exposedTo",
            })

    # Write nodes.csv
    node_fields = [":ID", ":LABEL", "ticker", "cik", "name", "text",
                   "objective_type", "status", "risk_category"]
    nodes_path = KG_DIR / "nodes.csv"
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_fields)
        writer.writeheader()
        writer.writerows(node_rows)
    print(f"\n✓ Wrote {len(node_rows)} nodes to {nodes_path}")

    # Write edges.csv
    edge_fields = [":START_ID", ":END_ID", ":TYPE"]
    edges_path = KG_DIR / "edges.csv"
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_fields)
        writer.writeheader()
        writer.writerows(edge_rows)
    print(f"✓ Wrote {len(edge_rows)} edges to {edges_path}")

    return node_rows, edge_rows


def build_all():
    """Main pipeline: build KG for all 6 companies."""
    KG_JSON_DIR.mkdir(parents=True, exist_ok=True)

    all_kg = {}

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"Building KG for {ticker}")
        print(f"{'='*50}")

        kg = build_kg_for_company(ticker)
        if kg:
            # Save per-company JSON
            json_path = KG_JSON_DIR / f"{ticker}_kg.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(kg, f, indent=2)
            print(f"  ✓ Saved {json_path.name}")
            all_kg[ticker] = kg
        else:
            print(f"  ✗ Skipped {ticker}")

    # Merge all into CSV
    if all_kg:
        merge_to_csv(all_kg)

    print(f"\n{'='*50}")
    print(f"KG BUILD COMPLETE: {len(all_kg)}/{len(TICKERS)} companies")
    print(f"{'='*50}")

    return all_kg


if __name__ == "__main__":
    build_all()
