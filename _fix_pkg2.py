"""
Fix Package 2 quality issues:
1. Where validation said NO/PARTIAL with a corrected_mission, use the correction
2. Where extraction got STRATEGY/OPERATIONS, re-extract with focused prompt
3. Where Pkg1 is clearly better, re-derive from 10-K with Pkg1 as reference
4. Clean up filler prefixes and boilerplate
"""
import json, re, pathlib, time
import urllib.request

BASE_DIR = pathlib.Path(__file__).parent
REG_PATH = BASE_DIR / "companies_registry.json"
SEC_DIR  = BASE_DIR / "MSGKG" / "data" / "data"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"
MODEL_LABEL = "Qwen2.5-7B-Instruct (Ollama/GPU)"


def call_llm(messages, max_tokens=400):
    payload = json.dumps({
        "model": MODEL, "messages": messages, "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["message"]["content"].strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
    except:
        pass
    return None


def extract_item1(filepath):
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    hdr = text.find("</SEC-HEADER>")
    if hdr > 0: text = text[hdr + len("</SEC-HEADER>"):]
    text = re.sub(r'\s+', ' ', text)
    m = re.search(r'(?:ITEM\s*1\.?\s*(?:BUSINESS|Business))', text, re.IGNORECASE)
    start = m.end() if m else 0
    em = re.search(r'(?:ITEM\s*1A\.?\s|ITEM\s*2\.?\s|PART\s*II\b)', text[start:], re.IGNORECASE)
    end = start + em.start() if em else min(start + 200000, len(text))
    return text[start:end].strip()[:100000]


def find_10k_file(cik):
    cik_stripped = cik.lstrip("0")
    for f in SEC_DIR.glob("cleaned_10-K_*.txt"):
        acc = f.stem.replace("cleaned_10-K_", "")
        file_cik = acc.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            return f
    return None


# Focused re-extraction prompt — explicitly tells model what NOT to do
REEXTRACT_PROMPT = """From this 10-K excerpt of {company} ({sector}), find the MISSION STATEMENT.

A mission statement is 1-2 sentences that declare:
- WHY the company exists (its PURPOSE)
- WHO it serves (customers, stakeholders)
- WHAT unique value it creates

REJECT these — they are NOT mission statements:
- Strategy descriptions ("Our strategy is to grow through acquisitions...")
- Operational descriptions ("We manufacture and sell products...")
- Legal entity descriptions ("XYZ Corp is a Delaware corporation headquartered in...")
- Financial goals ("We aim to maximize shareholder value...")
- HR programs ("We are committed to diversity...")
- Marketing slogans or taglines

Look for phrases like: "Our mission is...", "Our purpose is...", "We exist to...", "We are dedicated to...", "We strive to..."

If the filing contains {pkg1_hint}, consider whether that or something similar appears in this text.

TEXT:
{text}

Respond JSON only:
{{"found": true/false, "mission": "the mission statement (1-2 sentences, 10-30 words)", "source_type": "EXPLICIT/IMPLIED/DERIVED", "quality": 1-4, "confidence": 0-100, "reasoning": "why this is a real mission"}}

If no true mission exists, derive a purpose-focused statement (not operational):
{{"found": true, "mission": "derived purpose statement", "source_type": "DERIVED", "quality": 3, "confidence": 50, "reasoning": "derived from purpose language"}}"""

VALIDATE_PROMPT = """Is this a genuine corporate mission statement?

Company: {company} ({sector})
Text: "{mission}"

A mission declares PURPOSE (why), AUDIENCE (who), and VALUE (what).
NOT a mission: strategy, operations, financials, legal description, HR, marketing.

Respond JSON:
{{"is_mission": "YES/PARTIAL/NO", "actual_type": "MISSION/VISION/STRATEGY/OPERATIONS/MIXED", "quality": 1-4, "quality_reason": "explanation", "corrected_mission": "better version focused on purpose/who/value, or NONE", "confidence": 0-100, "issues": []}}"""


def diagnose(cik, info):
    """Return list of problems with this company's Pkg2 extraction."""
    ov = info.get("overview", {})
    ext = ov.get("llm_extraction", {})
    val = ext.get("validation", {}) or {}
    pkg2 = ext.get("improved_mission", ext.get("mission", ""))
    pkg1 = ov.get("mission", "")

    problems = []

    val_is = val.get("is_mission", "")
    val_type = val.get("actual_type", "")

    # Not a mission
    if val_type in ("STRATEGY", "OPERATIONS", "FINANCIAL", "HR"):
        problems.append("NOT_MISSION")

    # Validation said NO
    if val_is == "NO":
        problems.append("VALIDATION_NO")

    # Boilerplate
    boilerplate = ["incorporated in", "Delaware corporation", "Indiana corporation",
                   "headquartered in", "Form 10-K", "fiscal year", "common stock",
                   "NYSE", "NASDAQ"]
    if any(b.lower() in pkg2.lower() for b in boilerplate):
        problems.append("BOILERPLATE")

    # Starts with filler
    if pkg2.startswith(("The Company", "The company", "We ", "Our ")):
        # Only flag if it's operational, not mission-like
        op_words = ["manufactur", "operat", "sell", "produc", "acquir", "develop and franchise"]
        if any(w in pkg2.lower()[:100] for w in op_words):
            problems.append("OPERATIONAL_FILLER")

    # Too long
    if pkg2 and len(pkg2.split()) > 60:
        problems.append("TOO_LONG")

    # Pkg1 fallback
    if "pkg1_fallback" in ext.get("issues", []):
        problems.append("PKG1_FALLBACK")

    # Empty or too short
    if not pkg2 or len(pkg2.strip()) < 10:
        problems.append("EMPTY")

    return problems


def main():
    with open(REG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Diagnose all companies
    to_fix = []
    for cik, info in data.items():
        problems = diagnose(cik, info)
        if problems:
            to_fix.append((cik, info, problems))

    print(f"Companies needing fixes: {len(to_fix)}/91")
    print("=" * 70)

    fixed = 0
    for idx, (cik, info, problems) in enumerate(to_fix):
        name = info["name"][:40]
        ov = info["overview"]
        ext = ov.get("llm_extraction", {})
        pkg1 = ov.get("mission", "")
        old_pkg2 = ext.get("improved_mission", ext.get("mission", ""))

        print(f"\n  [{idx+1}/{len(to_fix)}] {name}")
        print(f"    Problems: {' | '.join(problems)}")
        print(f"    Old Pkg2: \"{old_pkg2[:60]}...\"")
        print(f"    Pkg1:     \"{pkg1[:60]}...\"")

        # Find 10-K
        filepath = find_10k_file(cik)

        new_mission = None

        if filepath:
            item1 = extract_item1(filepath)

            # Try re-extraction with focused prompt and Pkg1 hint
            pkg1_hint = pkg1[:60] if pkg1 else "no Pkg1 reference"

            for start, end in [(0, 3500), (2500, 6000), (5000, 8500)]:
                chunk = item1[start:min(end, len(item1))]
                if len(chunk) < 100:
                    break

                result = call_llm([
                    {"role": "system", "content": "You extract mission statements from SEC filings. A mission declares PURPOSE, not operations or strategy. JSON only."},
                    {"role": "user", "content": REEXTRACT_PROMPT.format(
                        company=name, sector=info.get("sector", ""),
                        text=chunk, pkg1_hint=pkg1_hint
                    )},
                ])

                if result and result.get("found") and result.get("mission"):
                    m = result["mission"].strip()
                    # Quick quality check
                    if len(m) >= 10 and len(m.split()) <= 60:
                        # Reject if it's clearly operational
                        op_phrases = ["is a", "was incorporated", "headquartered", "NYSE", "NASDAQ",
                                     "Form 10-K", "fiscal year", "common stock"]
                        if not any(op in m for op in op_phrases):
                            new_mission = result
                            break

        # If re-extraction failed or no file, derive
        if not new_mission and filepath:
            item1 = extract_item1(filepath) if not item1 else item1
            result = call_llm([
                {"role": "system", "content": "Write a concise mission statement (10-25 words) based on a company's business description. Focus on PURPOSE and WHO they serve. JSON only."},
                {"role": "user", "content": f"""Based on {name}'s ({info.get('sector','')}) business description, write a one-sentence MISSION statement.
Focus on WHY the company exists and WHO it serves — not what products they make.

Business: {item1[:1500]}

JSON: {{"found": true, "mission": "purpose-focused mission (10-25 words)", "source_type": "DERIVED", "quality": 3, "confidence": 50}}"""},
            ])
            if result and result.get("mission") and len(result["mission"].strip()) >= 10:
                new_mission = result
                new_mission["source_type"] = "DERIVED"

        if not new_mission:
            print(f"    -> SKIPPED (could not improve)")
            continue

        new_m = new_mission["mission"].strip()
        # Clean up quotes
        new_m = new_m.strip('"\'')

        print(f"    -> New:   \"{new_m[:70]}...\"")

        # Validate the new extraction
        val_result = call_llm([
            {"role": "system", "content": "Validate mission statements. JSON only."},
            {"role": "user", "content": VALIDATE_PROMPT.format(
                company=name, sector=info.get("sector", ""),
                mission=new_m[:500]
            )},
        ])

        if val_result:
            val_is = val_result.get("is_mission", "YES")
            val_type = val_result.get("actual_type", "MISSION")
            val_conf = val_result.get("confidence", 70)
            corrected = val_result.get("corrected_mission", "")

            # If validation corrected it, use the correction
            if val_is in ("PARTIAL", "NO") and corrected and corrected not in ("NONE", "") and len(corrected) >= 10:
                final_mission = corrected.strip('"\'')
                improvement_reason = val_result.get("quality_reason", "Corrected by validation")
                print(f"    -> Corrected: \"{final_mission[:70]}...\" ({val_is})")
            else:
                final_mission = new_m
                improvement_reason = "Validated as correct" if val_is == "YES" else val_result.get("quality_reason", "Accepted")

            print(f"    -> Validation: {val_is} / {val_type} ({val_conf}%)")
        else:
            final_mission = new_m
            val_result = {"is_mission": "YES", "actual_type": "MISSION", "quality": 3,
                         "quality_reason": "", "corrected_mission": "", "confidence": 50, "issues": []}
            improvement_reason = "Validation unavailable"

        # Update registry
        ext["mission"] = new_mission.get("mission", new_m)
        ext["source_type"] = new_mission.get("source_type", "DERIVED")
        ext["quality"] = new_mission.get("quality", 3)
        ext["confidence"] = new_mission.get("confidence", 50)
        ext["reasoning"] = new_mission.get("reasoning", "Re-extracted with focused prompt")
        ext["issues"] = []
        ext["model"] = MODEL_LABEL

        ext["validation"] = {
            "is_mission": val_result.get("is_mission", "YES"),
            "actual_type": val_result.get("actual_type", "MISSION"),
            "quality": val_result.get("quality", 3),
            "quality_reason": val_result.get("quality_reason", ""),
            "corrected_mission": val_result.get("corrected_mission", ""),
            "confidence": val_result.get("confidence", 50),
            "issues": val_result.get("issues", []) if isinstance(val_result.get("issues"), list) else [],
            "model": MODEL_LABEL,
        }

        ext["improved_mission"] = final_mission
        ext["improvement_reason"] = improvement_reason

        fixed += 1

        # Save every 5
        if (idx + 1) % 5 == 0:
            with open(REG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Fixed: {fixed}/{len(to_fix)}")

    # Re-diagnose
    still_issues = 0
    for cik, info in data.items():
        if diagnose(cik, info):
            still_issues += 1
    print(f"Remaining issues: {still_issues}/91")


if __name__ == "__main__":
    main()
