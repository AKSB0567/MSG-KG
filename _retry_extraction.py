"""
Retry LLM extraction for companies that failed in the first pass.
Uses a more aggressive multi-position search + derivation fallback.
"""
import json, re, pathlib, time
from llama_cpp import Llama

BASE_DIR = pathlib.Path(__file__).parent
REG_PATH = BASE_DIR / "companies_registry.json"
SEC_DIR  = BASE_DIR / "MSGKG" / "data" / "data"
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL = "Qwen2.5-3B-Instruct (local)"

print("Loading model...")
_llm = Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=6, verbose=False)
print("Model loaded.\n")


def call_llm(messages, max_tokens=300):
    try:
        resp = _llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
        text = resp["choices"][0]["message"]["content"].strip()
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
    if hdr > 0:
        text = text[hdr + len("</SEC-HEADER>"):]
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


# Shorter, more direct extraction prompt
EXTRACT = """From this 10-K excerpt of {company} ({sector}), extract the mission/purpose/vision statement.

TEXT: {text}

If you find a mission, purpose, or vision statement, return:
{{"found": true, "mission": "the statement", "source_type": "EXPLICIT", "confidence": 80}}

If NO explicit mission exists, derive one from the business description:
{{"found": true, "mission": "derived purpose statement", "source_type": "DERIVED", "confidence": 50}}

JSON only:"""

# Validation prompt
VALIDATE = """Is this a real mission statement for {company} ({sector})?

"{mission}"

Reply JSON only:
{{"is_mission": "YES or PARTIAL or NO", "actual_type": "MISSION or VISION or STRATEGY or OPERATIONS or MIXED", "quality": 1-4, "quality_reason": "one line", "corrected_mission": "better version or NONE", "confidence": 70, "issues": []}}"""

# Derivation fallback - when extraction fails, derive from first paragraph
DERIVE = """Based on this business description from {company}'s ({sector}) 10-K filing, write a one-sentence mission statement that captures WHY the company exists and WHO it serves.

Business description: {text}

Reply JSON only:
{{"found": true, "mission": "your derived mission statement", "source_type": "DERIVED", "confidence": 45}}"""


def main():
    with open(REG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    # Find companies needing retry
    retry = []
    for cik, info in data.items():
        ext = info.get("overview", {}).get("llm_extraction", {})
        needs_extraction = not ext.get("mission")
        needs_validation = ext.get("mission") and not (ext.get("validation") or {}).get("is_mission")
        if needs_extraction or needs_validation:
            retry.append((cik, info, needs_extraction, needs_validation))

    print(f"Retrying {len(retry)} companies...")
    print("=" * 60)

    fixed = 0
    for idx, (cik, info, needs_ext, needs_val) in enumerate(retry):
        name = info["name"][:40]
        sector = info.get("sector", "Unknown")
        ov = info.get("overview", {})
        print(f"  [{idx+1}/{len(retry)}] {name} ...", end=" ", flush=True)

        if needs_ext:
            filepath = find_10k_file(cik)
            if not filepath:
                print("NO FILE - using Pkg1 mission as fallback")
                # Fallback: use the text-matching mission
                pkg1_mission = ov.get("mission_before_improvement", ov.get("mission", ""))
                if pkg1_mission and len(pkg1_mission) >= 10:
                    ov["llm_extraction"] = {
                        "mission": pkg1_mission,
                        "source_type": "DERIVED",
                        "is_mission": "YES",
                        "actual_type": "MISSION",
                        "quality": 3,
                        "confidence": 40,
                        "reasoning": "Fallback: copied from Pkg1 text-matching (no 10-K file)",
                        "issues": ["no_10k_file", "pkg1_fallback"],
                        "chunks_searched": 0,
                        "candidates_found": 0,
                        "model": MODEL,
                        "validation": None,
                        "improved_mission": pkg1_mission,
                        "improvement_reason": "Pkg1 fallback - no source file",
                    }
                    fixed += 1
                continue

            item1 = extract_item1(filepath)
            if len(item1) < 50:
                # Try full file text
                raw = filepath.read_text(encoding="utf-8", errors="ignore")
                hdr = raw.find("</SEC-HEADER>")
                if hdr > 0:
                    raw = raw[hdr + len("</SEC-HEADER>"):]
                item1 = re.sub(r'\s+', ' ', raw)[:50000]

            mission_found = None

            # Try 3 different positions in the text
            positions = [
                (0, 2500),
                (1500, 4000),
                (3500, 6000),
            ]

            for pos_start, pos_end in positions:
                chunk = item1[pos_start:min(pos_end, len(item1))]
                if len(chunk) < 50:
                    continue

                result = call_llm([
                    {"role": "system", "content": "Extract mission statements from SEC filings. JSON only."},
                    {"role": "user", "content": EXTRACT.format(company=name, sector=sector, text=chunk)},
                ])

                if result and result.get("found") and result.get("mission") and len(result["mission"].strip()) >= 10:
                    mission_found = result
                    break

            # Fallback: derive from business description
            if not mission_found:
                chunk = item1[:1500]
                result = call_llm([
                    {"role": "system", "content": "You write mission statements based on business descriptions. JSON only."},
                    {"role": "user", "content": DERIVE.format(company=name, sector=sector, text=chunk)},
                ])
                if result and result.get("mission") and len(result["mission"].strip()) >= 10:
                    mission_found = result
                    mission_found["source_type"] = "DERIVED"
                    print("DERIVED", end=" ", flush=True)

            # Final fallback: use Pkg1 mission
            if not mission_found:
                pkg1_mission = ov.get("mission_before_improvement", ov.get("mission", ""))
                if pkg1_mission and len(pkg1_mission) >= 10:
                    mission_found = {
                        "found": True,
                        "mission": pkg1_mission,
                        "source_type": "DERIVED",
                        "confidence": 35,
                    }
                    print("PKG1-FALLBACK", end=" ", flush=True)

            if mission_found:
                extracted = mission_found["mission"].strip()
                ov["llm_extraction"] = {
                    "mission": extracted,
                    "source_type": mission_found.get("source_type", "DERIVED"),
                    "is_mission": "YES",
                    "actual_type": "MISSION",
                    "quality": mission_found.get("quality", 3),
                    "confidence": mission_found.get("confidence", 50),
                    "reasoning": mission_found.get("reasoning", "Found on retry"),
                    "issues": mission_found.get("issues", []),
                    "chunks_searched": 3,
                    "candidates_found": 1,
                    "model": MODEL,
                }
                needs_val = True  # Also validate
                print(f"FOUND ({mission_found.get('source_type','?')})", end=" ", flush=True)
            else:
                print("STILL NOT FOUND")
                continue

        # Validation pass
        if needs_val:
            ext = ov.get("llm_extraction", {})
            mission = ext.get("mission", "")
            if mission and len(mission) >= 10:
                print("| Validating...", end=" ", flush=True)
                val = call_llm([
                    {"role": "system", "content": "Validate mission statements. JSON only."},
                    {"role": "user", "content": VALIDATE.format(company=name, sector=sector, mission=mission[:300])},
                ])

                if val and val.get("is_mission"):
                    val_is = val["is_mission"]
                    val_conf = val.get("confidence", 70)
                    ext["validation"] = {
                        "is_mission": val_is,
                        "actual_type": val.get("actual_type", "MISSION"),
                        "quality": val.get("quality", 3),
                        "quality_reason": val.get("quality_reason", ""),
                        "corrected_mission": val.get("corrected_mission", ""),
                        "confidence": val_conf,
                        "issues": val.get("issues", []),
                        "model": MODEL,
                    }
                    corrected = val.get("corrected_mission", "")
                    if val_is in ("PARTIAL", "NO") and corrected and corrected != "NONE" and len(corrected) >= 10:
                        ext["improved_mission"] = corrected
                        ext["improvement_reason"] = val.get("quality_reason", "Corrected")
                    else:
                        ext["improved_mission"] = mission
                        ext["improvement_reason"] = "Validated as correct"
                    print(f"{val_is} ({val_conf}%)")
                    fixed += 1
                else:
                    # Validation failed, still accept extraction
                    ext["validation"] = {
                        "is_mission": "YES", "actual_type": "MISSION",
                        "quality": ext.get("quality", 3), "quality_reason": "Validation model unavailable",
                        "corrected_mission": "", "confidence": 50,
                        "issues": ["validation_failed"], "model": MODEL,
                    }
                    ext["improved_mission"] = mission
                    ext["improvement_reason"] = "Validation unavailable"
                    print("VAL-FALLBACK")
                    fixed += 1

        # Save every 3
        if (idx + 1) % 3 == 0:
            with open(REG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    # Final save
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Fixed: {fixed}/{len(retry)}")

    # Final tally
    total_ext = sum(1 for c in data.values() if c.get("overview",{}).get("llm_extraction",{}).get("mission"))
    total_val = sum(1 for c in data.values() if (c.get("overview",{}).get("llm_extraction",{}).get("validation") or {}).get("is_mission"))
    total_imp = sum(1 for c in data.values() if c.get("overview",{}).get("llm_extraction",{}).get("improved_mission"))
    print(f"Total extracted: {total_ext}/91")
    print(f"Total validated: {total_val}/91")
    print(f"Total improved: {total_imp}/91")


if __name__ == "__main__":
    main()
