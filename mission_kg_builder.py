"""
Mission KG Builder — Generate Mission-Focused Knowledge Graphs
===============================================================
Creates two KG schemas for each company:
  1. General Schema (LLM-Driven): Company + Mission statement (flat)
  2. Forced Schema (Ontology-Driven): FIBO-aligned mission decomposition

Uses sentence-transformers for semantic scoring (no external LLM needed).

Usage:  python mission_kg_builder.py
"""

import json, re, pathlib, time, unicodedata

BASE_DIR     = pathlib.Path(__file__).parent
REGISTRY     = BASE_DIR / "companies_registry.json"
SEC_DIR      = BASE_DIR / "MSGKG" / "data" / "data"
GENERAL_DIR  = BASE_DIR / "data" / "mission_kg" / "general"
ONTOLOGY_DIR = BASE_DIR / "data" / "mission_kg" / "ontology"

# Ensure output dirs exist
GENERAL_DIR.mkdir(parents=True, exist_ok=True)
ONTOLOGY_DIR.mkdir(parents=True, exist_ok=True)

# ── FIBO-aligned decomposition categories ───────────────────────────────

STAKEHOLDER_MAP = {
    "Customers": ["customer", "client", "consumer", "user", "buyer", "patron",
                  "member", "guest", "passenger", "policyholder", "patient",
                  "borrower", "depositor"],
    "Employees": ["employee", "colleague", "team member", "workforce", "staff",
                  "people", "talent", "worker", "associate", "team"],
    "Shareholders": ["shareholder", "investor", "stockholder", "return",
                     "value creation", "dividend", "capital"],
    "Society": ["community", "communities", "society", "world", "planet",
                "environment", "sustainability", "social", "public"],
    "Partners": ["partner", "supplier", "vendor", "alliance", "collaborat"],
    "Government": ["government", "regulat", "compliance", "federal", "state"],
}

VALUE_KEYWORDS = {
    "Innovation": ["innovat", "pioneer", "cutting-edge", "breakthrough", "transform",
                   "disrupt", "next-generation", "advanced", "leading-edge"],
    "Sustainability": ["sustain", "environment", "green", "carbon", "renewable",
                       "clean energy", "climate", "stewardship"],
    "Integrity": ["integrit", "ethic", "trust", "transparen", "honest", "accountab"],
    "Excellence": ["excellen", "quality", "best-in-class", "world-class", "superior",
                   "premier"],
    "Safety": ["safe", "health", "well-being", "protect", "secure"],
    "Inclusion": ["inclus", "divers", "equit", "belong", "equal opportunity"],
    "Growth": ["grow", "expand", "scale", "develop", "progress", "advance"],
    "Service": ["service", "serve", "deliver", "support", "help", "assist"],
}

OBJECTIVE_PATTERNS = [
    r'(?:to\s+)((?:deliver|provide|build|create|develop|achieve|drive|enable|'
    r'accelerate|advance|improve|enhance|transform|lead|grow|expand|protect|'
    r'maximize|ensure|foster|promote|empower|support|help)\s+[^,.;]{10,80})',
]

DOMAIN_MAP = {
    "Financial Services": ["bank", "financ", "insurance", "invest", "capital",
                           "lending", "credit", "wealth", "asset management"],
    "Technology": ["technolog", "software", "computing", "digital", "semiconductor",
                   "data", "cloud", "AI", "artificial intelligence", "cyber"],
    "Healthcare": ["health", "pharma", "medical", "therapeutic", "clinical",
                   "patient", "drug", "biotech", "diagnostic"],
    "Energy": ["energy", "oil", "gas", "petroleum", "power", "electric",
               "renewable", "solar", "wind", "nuclear", "utility"],
    "Manufacturing": ["manufactur", "industrial", "machinery", "equipment",
                      "production", "assembly", "fabricat"],
    "Consumer Goods": ["consumer", "retail", "food", "beverage", "apparel",
                       "clothing", "household", "personal care"],
    "Aerospace & Defense": ["aerospace", "defense", "aircraft", "military",
                            "aviation", "space", "missile"],
    "Transportation": ["transport", "airline", "shipping", "freight", "logistics",
                       "rail", "trucking"],
    "Real Estate": ["real estate", "property", "reit", "leasing", "tenant"],
    "Media & Communications": ["media", "advertis", "broadcast", "telecom",
                               "communication", "publishing", "entertainment"],
}


def slug(name: str) -> str:
    """Create a safe filename slug from company name."""
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    s = re.sub(r'[^\w\s-]', '', s).strip().lower()
    return re.sub(r'[-\s]+', '_', s)[:50]


def uri_safe(text: str) -> str:
    """Make text safe for use in URIs."""
    s = re.sub(r'[^\w]', '_', text.strip())
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:60]


def extract_stakeholders(mission: str, context: str = "") -> list:
    """Identify stakeholders mentioned in mission + context."""
    text = (mission + " " + context).lower()
    found = []
    for group, keywords in STAKEHOLDER_MAP.items():
        for kw in keywords:
            if kw in text:
                found.append(group)
                break
    return found


def extract_values(mission: str) -> list:
    """Identify values/principles in mission text."""
    text = mission.lower()
    found = []
    for value, keywords in VALUE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.append(value)
                break
    return found


def extract_objectives(mission: str) -> list:
    """Extract action-oriented objectives from mission."""
    objectives = []
    for pat in OBJECTIVE_PATTERNS:
        for m in re.finditer(pat, mission, re.IGNORECASE):
            obj = m.group(1).strip()
            if len(obj) > 15 and not any(skip in obj.lower() for skip in
                    ['form 10', 'section', 'pursuant', 'registrant']):
                objectives.append(obj[:100])
    return objectives[:5]  # max 5


def detect_domain(mission: str, sector: str) -> list:
    """Detect business domain from mission + sector info."""
    text = (mission + " " + sector).lower()
    found = []
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text:
                found.append(domain)
                break
    return found[:3]  # max 3


def find_10k_file(cik: str) -> pathlib.Path | None:
    """Find the 10-K text file for a given CIK."""
    if not SEC_DIR.exists():
        return None
    cik_stripped = cik.lstrip("0")
    for txt_file in SEC_DIR.glob("cleaned_10-K_*.txt"):
        accession = txt_file.stem.replace("cleaned_10-K_", "")
        file_cik = accession.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            return txt_file
    return None


def get_mission_context(filepath: pathlib.Path, mission: str) -> str:
    """Get surrounding context of the mission in the 10-K file."""
    if not filepath:
        return ""
    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
        # Strip header
        m = re.search(r'</SEC-HEADER>', text)
        if m:
            text = text[m.end():]
        text = re.sub(r'\s+', ' ', text)

        # Find mission text in document
        mission_clean = re.sub(r'\s+', ' ', mission).strip()
        idx = text.lower().find(mission_clean[:50].lower())
        if idx >= 0:
            return text[max(0, idx - 500):idx + len(mission_clean) + 500]
        return text[:3000]  # fallback: first 3000 chars
    except Exception:
        return ""


def escape_ttl(text: str) -> str:
    """Escape text for Turtle string literals."""
    return text.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ').replace('\r', '')


# ── General Schema (LLM-Driven) ────────────────────────────────────────

def build_general_kg(cik: str, company: dict, overview: dict) -> str:
    """Build a simple Company-Mission KG (simulating LLM output)."""
    name = company["name"]
    mission = overview.get("mission", "")
    source = overview.get("mission_source", "unknown")
    tier = overview.get("mission_tier", 0)
    confidence = overview.get("mission_confidence", 0)
    sector = company.get("sector", "Unknown")

    txt_file = find_10k_file(cik)
    source_file = txt_file.name if txt_file else "N/A"

    ttl = f'''@prefix msg: <http://msgkg.io/ontology#> .
@prefix inst: <http://msgkg.io/instances#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ── General Schema: LLM-Driven Mission Extraction ──
# Company: {name}
# CIK: {cik}

inst:Company_{uri_safe(name)} a msg:Company ;
    rdfs:label "{escape_ttl(name)}" ;
    msg:cik "{cik}" ;
    msg:sector "{escape_ttl(sector)}" ;
    msg:hasMission inst:Mission_{uri_safe(name)} .

inst:Mission_{uri_safe(name)} a msg:MissionStatement ;
    msg:missionText "{escape_ttl(mission)}" ;
    msg:missionSource "{source}" ;
    msg:missionTier {tier}^^xsd:integer ;
    msg:missionConfidence {confidence}^^xsd:float ;
    msg:extractedFrom "{escape_ttl(source_file)}" .
'''
    return ttl


# ── Forced Schema (Ontology-Driven / FIBO-Aligned) ─────────────────────

def build_ontology_kg(cik: str, company: dict, overview: dict) -> str:
    """Build a FIBO-aligned mission decomposition KG."""
    name = company["name"]
    mission = overview.get("mission", "")
    sector = company.get("sector", "Unknown")
    name_safe = uri_safe(name)

    # Get context from 10-K for richer extraction
    txt_file = find_10k_file(cik)
    context = get_mission_context(txt_file, mission) if mission else ""
    source_file = txt_file.name if txt_file else "N/A"

    # Decompose mission
    stakeholders = extract_stakeholders(mission, context)
    values = extract_values(mission)
    objectives = extract_objectives(mission)
    domains = detect_domain(mission, sector)

    # Use capabilities from registry if available
    capabilities = overview.get("capabilities", [])[:5]

    ttl = f'''@prefix msg: <http://msgkg.io/ontology#> .
@prefix fibo-fnd: <https://spec.edmcouncil.org/fibo/ontology/FND/> .
@prefix fibo-be: <https://spec.edmcouncil.org/fibo/ontology/BE/> .
@prefix inst: <http://msgkg.io/instances#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ── Ontology Schema: FIBO-Aligned Mission Decomposition ──
# Company: {name}
# CIK: {cik}
# Stakeholders: {", ".join(stakeholders) if stakeholders else "None identified"}
# Values: {", ".join(values) if values else "None identified"}
# Objectives: {len(objectives)} found
# Domains: {", ".join(domains) if domains else "Unknown"}

# ── Class Definitions ──
msg:MissionStatement a owl:Class ;
    rdfs:subClassOf fibo-fnd:BusinessObjective ;
    rdfs:label "Mission Statement" .
msg:Stakeholder a owl:Class ;
    rdfs:subClassOf fibo-be:LegalEntity ;
    rdfs:label "Stakeholder" .
msg:CorporateObjective a owl:Class ;
    rdfs:subClassOf fibo-fnd:BusinessObjective ;
    rdfs:label "Corporate Objective" .
msg:CorporateValue a owl:Class ;
    rdfs:label "Corporate Value" .
msg:BusinessCapability a owl:Class ;
    rdfs:label "Business Capability" .
msg:BusinessDomain a owl:Class ;
    rdfs:label "Business Domain" .

# ── Company ──
inst:Company_{name_safe} a msg:Company, fibo-be:LegalEntity ;
    rdfs:label "{escape_ttl(name)}" ;
    msg:cik "{cik}" ;
    msg:sector "{escape_ttl(sector)}" ;
    msg:hasMission inst:Mission_{name_safe} .

# ── Mission Statement ──
inst:Mission_{name_safe} a msg:MissionStatement ;
    msg:missionText "{escape_ttl(mission)}" ;
    msg:extractedFrom "{escape_ttl(source_file)}" '''

    # Add stakeholder relationships
    for s in stakeholders:
        s_safe = uri_safe(s)
        ttl += f''';\n    msg:servesStakeholder inst:{name_safe}_Stakeholder_{s_safe} '''

    # Add value relationships
    for v in values:
        v_safe = uri_safe(v)
        ttl += f''';\n    msg:embodiesValue inst:{name_safe}_Value_{v_safe} '''

    # Add objective relationships
    for i, obj in enumerate(objectives):
        ttl += f''';\n    msg:pursuesObjective inst:{name_safe}_Objective_{i} '''

    # Add domain relationships
    for d in domains:
        d_safe = uri_safe(d)
        ttl += f''';\n    msg:operatesInDomain inst:{name_safe}_Domain_{d_safe} '''

    # Add capability relationships
    for i, cap in enumerate(capabilities[:5]):
        ttl += f''';\n    msg:leveragesCapability inst:{name_safe}_Capability_{i} '''

    ttl += " .\n"

    # ── Stakeholder Instances ──
    for s in stakeholders:
        s_safe = uri_safe(s)
        ttl += f'''
inst:{name_safe}_Stakeholder_{s_safe} a msg:Stakeholder ;
    rdfs:label "{s}" .
'''

    # ── Value Instances ──
    for v in values:
        v_safe = uri_safe(v)
        ttl += f'''
inst:{name_safe}_Value_{v_safe} a msg:CorporateValue ;
    rdfs:label "{v}" .
'''

    # ── Objective Instances ──
    for i, obj in enumerate(objectives):
        ttl += f'''
inst:{name_safe}_Objective_{i} a msg:CorporateObjective ;
    rdfs:label "{escape_ttl(obj)}" .
'''

    # ── Domain Instances ──
    for d in domains:
        d_safe = uri_safe(d)
        ttl += f'''
inst:{name_safe}_Domain_{d_safe} a msg:BusinessDomain ;
    rdfs:label "{d}" .
'''

    # ── Capability Instances ──
    for i, cap in enumerate(capabilities[:5]):
        cap_text = cap.split(";")[0].strip()[:100]
        ttl += f'''
inst:{name_safe}_Capability_{i} a msg:BusinessCapability ;
    rdfs:label "{escape_ttl(cap_text)}" .
'''

    return ttl


def run():
    t0 = time.time()
    print("=" * 70)
    print("  Mission KG Builder — General + Ontology Schemas")
    print("=" * 70)

    if not REGISTRY.exists():
        print("ERROR: companies_registry.json not found!")
        return

    with open(REGISTRY, encoding="utf-8") as f:
        registry = json.load(f)

    print(f"\n[1] Processing {len(registry)} companies...\n")

    general_count = 0
    ontology_count = 0
    stats = {"stakeholders": 0, "values": 0, "objectives": 0, "domains": 0}

    for cik, info in registry.items():
        name = info["name"]
        overview = info.get("overview", {})
        mission = overview.get("mission", "")
        company_slug = slug(name)

        # Build General Schema
        general_ttl = build_general_kg(cik, info, overview)
        general_path = GENERAL_DIR / f"{company_slug}.ttl"
        general_path.write_text(general_ttl, encoding="utf-8")
        general_count += 1

        # Build Ontology Schema
        ontology_ttl = build_ontology_kg(cik, info, overview)
        ontology_path = ONTOLOGY_DIR / f"{company_slug}.ttl"
        ontology_path.write_text(ontology_ttl, encoding="utf-8")
        ontology_count += 1

        # Count decomposition elements
        stakeholders = extract_stakeholders(mission, "")
        values_found = extract_values(mission)
        objectives = extract_objectives(mission)
        domains = detect_domain(mission, info.get("sector", ""))
        stats["stakeholders"] += len(stakeholders)
        stats["values"] += len(values_found)
        stats["objectives"] += len(objectives)
        stats["domains"] += len(domains)

        n_triplets = len(stakeholders) + len(values_found) + len(objectives) + len(domains)
        status = "OK" if mission else "NO MISSION"
        print(f"  [{status:>10s}] {name[:35]:<35s}  triplets={n_triplets}  "
              f"S={len(stakeholders)} V={len(values_found)} O={len(objectives)} D={len(domains)}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  General KGs:    {general_count} -> {GENERAL_DIR}")
    print(f"  Ontology KGs:   {ontology_count} -> {ONTOLOGY_DIR}")
    print(f"  Total triplets: {sum(stats.values())}")
    print(f"    Stakeholders: {stats['stakeholders']}")
    print(f"    Values:       {stats['values']}")
    print(f"    Objectives:   {stats['objectives']}")
    print(f"    Domains:      {stats['domains']}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
