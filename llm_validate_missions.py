"""
LLM-based Mission Statement Validation Pipeline
Uses Qwen2.5-72B-Instruct via HuggingFace Inference API
"""
import json, pathlib, re, time
from huggingface_hub import InferenceClient

# ── Config ──
REG_PATH = pathlib.Path("companies_registry.json")
SEC_DIR = pathlib.Path("MSGKG/data/data")
MODEL = "Qwen/Qwen2.5-72B-Instruct"

# ── Load data ──
with open(REG_PATH, encoding="utf-8") as f:
    data = json.load(f)

client = InferenceClient(provider="novita")

PROMPT_TEMPLATE = """You are a financial analyst validating corporate mission statements extracted from SEC 10-K filings.

Company: {company_name}
Sector: {sector}

Extracted text claimed to be the mission statement:
"{mission}"

Context from the 10-K filing (text surrounding the extracted mission):
"{context}"

Evaluate this extraction by answering EXACTLY in this JSON format (no other text):
{{
  "is_mission": "YES" or "NO" or "PARTIAL",
  "actual_type": "MISSION" or "VISION" or "STRATEGY" or "OPERATIONS" or "HR" or "FINANCIAL" or "MIXED",
  "mission_quality": 1 or 2 or 3 or 4 or 0,
  "quality_reason": "one sentence explanation",
  "corrected_mission": "if PARTIAL or NO, suggest the actual mission portion or state NONE",
  "confidence": 0 to 100,
  "issues": ["list of issues found, e.g. 'contains strategic metrics', 'HR language', 'truncated sentence'"]
}}

Mission quality codes:
- 1 = Directly stated as mission/purpose/vision in the filing
- 2 = Present in filing but labeled as commitment/goal/belief, requires inference
- 3 = Not explicitly stated, derived from business description context
- 4 = Vague or generic corporate language
- 0 = Not a mission statement at all

Be strict. A real mission statement declares WHY the company exists and WHO it serves. Strategy (how to grow), operations (what they do), HR (employee programs), and financial goals are NOT missions."""


def get_context(cik, mission):
    """Get surrounding context from 10-K file."""
    cik_stripped = cik.lstrip("0")
    for f in SEC_DIR.glob("cleaned_10-K_*.txt"):
        acc = f.stem.replace("cleaned_10-K_", "")
        file_cik = acc.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            text = f.read_text(encoding="utf-8", errors="ignore")
            hdr = text.find("</SEC-HEADER>")
            if hdr > 0:
                text = text[hdr + len("</SEC-HEADER>"):]
            text_norm = re.sub(r"\s+", " ", text)
            text_lower = text_norm.lower()

            mission_lower = mission.lower()[:50]
            idx = text_lower.find(mission_lower)
            if idx >= 0:
                start = max(0, idx - 500)
                end = min(len(text_norm), idx + len(mission) + 500)
                return text_norm[start:end]

            # Keyword fallback
            words = set(re.findall(r"\b[a-z]{4,}\b", mission.lower())) - {
                "the", "and", "for", "that", "with", "from", "are", "our",
                "has", "have", "been", "will", "their", "this", "not", "but",
            }
            best_score, best_pos = 0, 0
            for i in range(0, min(len(text_lower), 300000), 300):
                chunk = text_lower[i:i+600]
                score = sum(1 for w in words if w in chunk)
                if score > best_score:
                    best_score = score
                    best_pos = i
            if best_score >= 3:
                return text_norm[max(0, best_pos-200):best_pos+800]

    return ""


def validate_company(cik, info):
    """Validate a single company's mission via LLM."""
    ov = info.get("overview", {})
    mission = ov.get("mission", "")
    name = info.get("name", "Unknown")
    sector = info.get("sector", "Unknown")

    if not mission or len(mission) < 10:
        return {
            "is_mission": "NO", "actual_type": "NONE",
            "mission_quality": 0, "quality_reason": "No mission extracted",
            "corrected_mission": "NONE", "confidence": 0, "issues": ["no_mission"]
        }

    context = get_context(cik, mission)
    if not context:
        context = ov.get("mission_evidence", "")

    prompt = PROMPT_TEMPLATE.format(
        company_name=name,
        sector=sector,
        mission=mission[:500],
        context=context[:1000]
    )

    try:
        response = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only, no other text."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )

        # Parse JSON from response
        text = response.choices[0].message.content.strip()
        # Find JSON block
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            result = json.loads(text[json_start:json_end])
            return result
        else:
            return {"error": "no_json", "raw": text[:200]}

    except json.JSONDecodeError as e:
        return {"error": f"json_parse: {str(e)[:80]}", "raw": text[:200]}
    except Exception as e:
        return {"error": str(e)[:100]}


def main():
    results = {}
    total = len(data)
    validated = 0
    errors = 0

    print(f"Validating {total} companies with {MODEL}...")
    print("=" * 60)

    for i, (cik, info) in enumerate(data.items()):
        name = info.get("name", "?")[:35]
        mission = info.get("overview", {}).get("mission", "")

        if not mission or len(mission) < 10:
            print(f"[{i+1}/{total}] {name}: SKIP (no mission)")
            continue

        print(f"[{i+1}/{total}] {name}...", end=" ", flush=True)

        result = validate_company(cik, info)

        if "error" in result:
            print(f"ERROR: {result.get('error', '')[:50]}")
            errors += 1
            results[cik] = result
        else:
            is_m = result.get("is_mission", "?")
            mq = result.get("mission_quality", "?")
            conf = result.get("confidence", 0)
            atype = result.get("actual_type", "?")
            print(f"{is_m} | M{mq} | {conf}% | {atype}")

            # Update registry
            ov = data[cik].get("overview", {})
            ov["llm_is_mission"] = result.get("is_mission", "")
            ov["llm_actual_type"] = result.get("actual_type", "")
            ov["llm_quality"] = result.get("mission_quality", 0)
            ov["llm_quality_reason"] = result.get("quality_reason", "")
            ov["llm_corrected_mission"] = result.get("corrected_mission", "")
            ov["llm_confidence"] = result.get("confidence", 0)
            ov["llm_issues"] = result.get("issues", [])

            # Auto-update mission_quality if LLM is confident
            if conf >= 60:
                old_mq = ov.get("mission_quality", 0)
                new_mq = result.get("mission_quality", old_mq)
                if old_mq != new_mq:
                    ov["mission_quality_prev"] = old_mq
                    ov["mission_quality"] = new_mq

            validated += 1
            results[cik] = result

        # Rate limit: ~1 request per second for free tier
        time.sleep(1.5)

    # Save updated registry
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save raw LLM results
    with open("_llm_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Report
    print("\n" + "=" * 60)
    print(f"Validated: {validated}/{total}")
    print(f"Errors: {errors}")

    # Distribution
    is_mission_counts = {"YES": 0, "NO": 0, "PARTIAL": 0}
    type_counts = {}
    mq_counts = {}
    for r in results.values():
        if "error" not in r:
            im = r.get("is_mission", "?")
            is_mission_counts[im] = is_mission_counts.get(im, 0) + 1
            at = r.get("actual_type", "?")
            type_counts[at] = type_counts.get(at, 0) + 1
            mq = r.get("mission_quality", 0)
            mq_counts[mq] = mq_counts.get(mq, 0) + 1

    print(f"\nIs Mission: {is_mission_counts}")
    print(f"Actual Type: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")
    print(f"Quality: {dict(sorted(mq_counts.items()))}")

    # Flag problematic ones
    print("\nFlagged for review:")
    for cik, r in results.items():
        if "error" not in r:
            if r.get("is_mission") == "NO":
                print(f"  NOT MISSION: {data[cik]['name'][:35]} -> {r.get('actual_type')} | {r.get('quality_reason','')[:60]}")
            elif r.get("is_mission") == "PARTIAL":
                print(f"  PARTIAL: {data[cik]['name'][:35]} -> {r.get('corrected_mission','')[:60]}")


if __name__ == "__main__":
    main()
