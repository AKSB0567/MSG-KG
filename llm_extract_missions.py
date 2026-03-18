"""
Full LLM Mission Extraction Pipeline (Package 2)
=================================================
Sends the Item 1 (Business) section of each 10-K filing to a local LLM
via Ollama (GPU-accelerated) and extracts mission statements.

Model: Qwen2.5-7B-Instruct via Ollama (CUDA GPU)
Pipeline: Extract → Validate → Improve (3-step flow)

Stores results in companies_registry.json under overview.llm_extraction.

Run:  python llm_extract_missions.py
"""

import json, re, pathlib, time, sys
import urllib.request

# ── Config ──────────────────────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent
REG_PATH   = BASE_DIR / "companies_registry.json"
SEC_DIR    = BASE_DIR / "MSGKG" / "data" / "data"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL      = "qwen2.5:7b"
MODEL_LABEL = "Qwen2.5-7B-Instruct (Ollama/GPU)"


def call_llm(messages, max_tokens=400):
    """Call Ollama API and parse JSON from response."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["message"]["content"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
    except (json.JSONDecodeError, KeyError):
        return None
    except Exception as e:
        print(f"  LLM ERROR: {str(e)[:80]}")
    return None


def extract_item1(filepath):
    """Extract Item 1 (Business) section from a cleaned 10-K file."""
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    hdr = text.find("</SEC-HEADER>")
    if hdr > 0:
        text = text[hdr + len("</SEC-HEADER>"):]
    text = re.sub(r'\s+', ' ', text)

    m = re.search(r'(?:ITEM\s*1\.?\s*(?:BUSINESS|Business))', text, re.IGNORECASE)
    start = m.end() if m else 0
    em = re.search(r'(?:ITEM\s*1A\.?\s|ITEM\s*2\.?\s|PART\s*II\b)', text[start:], re.IGNORECASE)
    end = start + em.start() if em else min(start + 200000, len(text))
    return text[start:end].strip()[:200000]


def find_10k_file(cik):
    """Find the cleaned 10-K file for a given CIK."""
    cik_stripped = cik.lstrip("0")
    for f in SEC_DIR.glob("cleaned_10-K_*.txt"):
        acc = f.stem.replace("cleaned_10-K_", "")
        file_cik = acc.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            return f
    return None


# ── Prompts ─────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = "You are an expert financial analyst. Extract mission statements from SEC 10-K filings. Always respond with valid JSON."

EXTRACT_PROMPT = """Analyze this excerpt from the Item 1 (Business) section of {company}'s SEC 10-K filing.
Sector: {sector}

TEXT:
\"\"\"{text}\"\"\"

TASK: Find and extract the company's mission statement, purpose statement, or vision statement.

A REAL mission statement declares:
- WHY the company exists (purpose)
- WHO it serves (stakeholders)
- WHAT value it delivers

NOT a mission statement:
- Operational descriptions ("We manufacture widgets")
- Financial goals ("We aim to increase revenue")
- Legal entity descriptions ("XYZ Corp is a Delaware corporation")
- HR/employee programs, strategy/growth plans, regulatory disclosures

Respond with this JSON:
{{"found": true/false, "mission": "exact quote from filing or empty string", "source_type": "EXPLICIT/IMPLIED/DERIVED/NONE", "is_mission": "YES/PARTIAL/NO", "actual_type": "MISSION/VISION/STRATEGY/OPERATIONS/MIXED/NONE", "quality": 0-4, "confidence": 0-100, "reasoning": "one sentence"}}

Quality: 1=Directly stated, 2=Commitment/goal language, 3=Derived from context, 4=Vague/generic, 0=Not found
Source: EXPLICIT="Our mission is...", IMPLIED="We are committed to...", DERIVED=Inferred from description"""

VALIDATE_SYSTEM = "You validate mission statements. Always respond with valid JSON."

VALIDATE_PROMPT = """Evaluate whether this extracted text is truly a corporate mission statement.

Company: {company} ({sector})
Extracted text: "{mission}"

A REAL mission statement declares WHY the company exists, WHO it serves, and WHAT value it delivers.
NOT a mission: operational descriptions, financial targets, legal boilerplate, strategy plans, HR programs.

Respond with this JSON:
{{"is_mission": "YES/PARTIAL/NO", "actual_type": "MISSION/VISION/STRATEGY/OPERATIONS/HR/FINANCIAL/MIXED", "quality": 0-4, "quality_reason": "one sentence explanation", "corrected_mission": "if PARTIAL/NO, extract or rewrite the true mission portion, or NONE", "confidence": 0-100, "issues": ["list of issues"]}}"""

DERIVE_SYSTEM = "You write concise mission statements based on business descriptions. Always respond with valid JSON."

DERIVE_PROMPT = """Based on this business description from {company}'s ({sector}) 10-K filing, write a one-sentence mission statement that captures WHY the company exists, WHO it serves, and WHAT value it delivers.

Business description:
\"\"\"{text}\"\"\"

The mission statement should be 10-30 words, specific to this company (not generic), and capture their core purpose.

Respond with this JSON:
{{"found": true, "mission": "your derived mission statement", "source_type": "DERIVED", "quality": 3, "confidence": 50, "reasoning": "derived from business description"}}"""


# ── Main pipeline ───────────────────────────────────────────────────────

def main():
    # Test Ollama connection
    try:
        test = call_llm([{"role": "user", "content": "Say hi. JSON: {\"ok\":true}"}], max_tokens=20)
        if not test:
            print("ERROR: Cannot connect to Ollama. Run: ollama serve")
            sys.exit(1)
    except:
        print("ERROR: Ollama not running. Start it first.")
        sys.exit(1)

    with open(REG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    done = skipped = errors = no_file = 0

    print(f"\nPackage 2: LLM Mission Extraction Pipeline")
    print(f"Model: {MODEL_LABEL}")
    print(f"Companies: {total}")
    print("=" * 70)

    for i, (cik, info) in enumerate(data.items()):
        name = info.get("name", "Unknown")[:40]
        sector = info.get("sector", "Unknown")
        ov = info.get("overview", {})

        # Skip if already extracted
        if ov.get("llm_extraction") and ov["llm_extraction"].get("mission"):
            done += 1
            print(f"  [{i+1}/{total}] {name} ... CACHED")
            continue

        # Find 10-K file
        filepath = find_10k_file(cik)
        if not filepath:
            no_file += 1
            print(f"  [{i+1}/{total}] {name} ... NO FILE", end="", flush=True)
            # Fallback to Pkg1 mission
            pkg1_m = ov.get("mission_before_improvement", ov.get("mission", ""))
            if pkg1_m and len(pkg1_m) >= 10:
                ov["llm_extraction"] = _build_fallback(pkg1_m, "No 10-K file")
                print(" | PKG1 FALLBACK")
            else:
                print()
            continue

        print(f"  [{i+1}/{total}] {name} ...", end=" ", flush=True)

        item1_text = extract_item1(filepath)
        if len(item1_text) < 100:
            skipped += 1
            print("SKIP (too short)")
            continue

        # ── STEP 1: Extraction (try 3 positions) ──
        mission_result = None
        chunks_searched = 0

        for start, end in [(0, 3000), (2000, 5000), (4000, 7000)]:
            chunk = item1_text[start:min(end, len(item1_text))]
            if len(chunk) < 100:
                break
            chunks_searched += 1

            result = call_llm([
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": EXTRACT_PROMPT.format(
                    company=name, sector=sector, text=chunk
                )},
            ])

            if result and result.get("found") and result.get("mission"):
                m = result["mission"].strip()
                if len(m) >= 10:
                    mission_result = result
                    break

        # Derivation fallback
        if not mission_result:
            result = call_llm([
                {"role": "system", "content": DERIVE_SYSTEM},
                {"role": "user", "content": DERIVE_PROMPT.format(
                    company=name, sector=sector, text=item1_text[:2000]
                )},
            ])
            if result and result.get("mission") and len(result["mission"].strip()) >= 10:
                mission_result = result
                mission_result["source_type"] = "DERIVED"
                print("DERIVED |", end=" ", flush=True)

        # Pkg1 fallback
        if not mission_result:
            pkg1_m = ov.get("mission_before_improvement", ov.get("mission", ""))
            if pkg1_m and len(pkg1_m) >= 10:
                ov["llm_extraction"] = _build_fallback(pkg1_m, "LLM could not extract; using Pkg1")
                print("PKG1-FALLBACK")
                done += 1
                if (i + 1) % 5 == 0:
                    _save(data)
                continue
            else:
                print("NOT FOUND")
                ov["llm_extraction"] = {
                    "mission": "", "source_type": "NONE", "is_mission": "NO",
                    "actual_type": "NONE", "quality": 0, "confidence": 0,
                    "reasoning": "No mission found", "issues": ["not_found"],
                    "chunks_searched": chunks_searched, "candidates_found": 0,
                    "model": MODEL_LABEL, "validation": None,
                    "improved_mission": "", "improvement_reason": "No mission",
                }
                done += 1
                if (i + 1) % 5 == 0:
                    _save(data)
                continue

        extracted = mission_result["mission"].strip()
        src = mission_result.get("source_type", "DERIVED")
        quality = mission_result.get("quality", 3)
        conf = mission_result.get("confidence", 50)

        print(f"FOUND | M{quality} | {conf}% | {src}", end="", flush=True)

        # Build extraction record
        ov["llm_extraction"] = {
            "mission": extracted,
            "source_type": src,
            "is_mission": mission_result.get("is_mission", "YES"),
            "actual_type": mission_result.get("actual_type", "MISSION"),
            "quality": quality,
            "confidence": conf,
            "reasoning": mission_result.get("reasoning", ""),
            "issues": mission_result.get("issues", []) if isinstance(mission_result.get("issues"), list) else [],
            "chunks_searched": chunks_searched,
            "candidates_found": 1,
            "model": MODEL_LABEL,
        }

        # ── STEP 2: Validation ──
        print(" | Validating...", end="", flush=True)
        val = call_llm([
            {"role": "system", "content": VALIDATE_SYSTEM},
            {"role": "user", "content": VALIDATE_PROMPT.format(
                company=name, sector=sector, mission=extracted[:500]
            )},
        ])

        if val and val.get("is_mission"):
            val_is = val["is_mission"]
            val_conf = val.get("confidence", 70)
            print(f" {val_is} ({val_conf}%)", end="", flush=True)

            ov["llm_extraction"]["validation"] = {
                "is_mission": val_is,
                "actual_type": val.get("actual_type", "MISSION"),
                "quality": val.get("quality", quality),
                "quality_reason": val.get("quality_reason", ""),
                "corrected_mission": val.get("corrected_mission", ""),
                "confidence": val_conf,
                "issues": val.get("issues", []) if isinstance(val.get("issues"), list) else [],
                "model": MODEL_LABEL,
            }

            # ── STEP 3: Improve ──
            corrected = val.get("corrected_mission", "")
            if val_is in ("PARTIAL", "NO") and corrected and corrected not in ("NONE", "") and len(corrected) >= 10:
                ov["llm_extraction"]["improved_mission"] = corrected
                ov["llm_extraction"]["improvement_reason"] = val.get("quality_reason", "Corrected by validation")
                print(" | Improved", end="", flush=True)
            else:
                ov["llm_extraction"]["improved_mission"] = extracted
                ov["llm_extraction"]["improvement_reason"] = "Validated as correct"
        else:
            print(" | Val-skip", end="", flush=True)
            ov["llm_extraction"]["validation"] = {
                "is_mission": "YES", "actual_type": "MISSION",
                "quality": quality, "quality_reason": "Validation unavailable",
                "corrected_mission": "", "confidence": 50,
                "issues": ["validation_skipped"], "model": MODEL_LABEL,
            }
            ov["llm_extraction"]["improved_mission"] = extracted
            ov["llm_extraction"]["improvement_reason"] = "Validation unavailable"

        print()
        done += 1

        # Save every 5 companies
        if (i + 1) % 5 == 0:
            _save(data)

    # Final save
    _save(data)

    # ── Report ──
    print("\n" + "=" * 70)
    print(f"Done: {done} | Skipped: {skipped} | Errors: {errors} | No File: {no_file}")

    ext_c = sum(1 for c in data.values() if c.get("overview", {}).get("llm_extraction", {}).get("mission"))
    val_c = sum(1 for c in data.values() if (c.get("overview", {}).get("llm_extraction", {}).get("validation") or {}).get("is_mission"))
    imp_c = sum(1 for c in data.values() if c.get("overview", {}).get("llm_extraction", {}).get("improved_mission"))
    print(f"\nExtracted: {ext_c}/{total} | Validated: {val_c}/{total} | Improved: {imp_c}/{total}")
    print(f"Registry saved to {REG_PATH}")


def _build_fallback(mission, reason):
    """Build a fallback llm_extraction dict from Pkg1 mission."""
    return {
        "mission": mission, "source_type": "DERIVED",
        "is_mission": "YES", "actual_type": "MISSION",
        "quality": 3, "confidence": 40,
        "reasoning": reason, "issues": ["pkg1_fallback"],
        "chunks_searched": 0, "candidates_found": 0,
        "model": MODEL_LABEL,
        "validation": {
            "is_mission": "YES", "actual_type": "MISSION",
            "quality": 3, "quality_reason": "Fallback from Pkg1 text-matching",
            "corrected_mission": "", "confidence": 40,
            "issues": ["pkg1_fallback"], "model": MODEL_LABEL,
        },
        "improved_mission": mission,
        "improvement_reason": "Pkg1 fallback",
    }


def _save(data):
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
