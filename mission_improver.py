"""
Mission Statement Improvement Pipeline
=======================================
Phase 1: Apply 31 LLM-corrected missions for PARTIAL cases
Phase 2: Re-extract missions for 47 NO cases using full Item 1 context via LLM
Phase 3: For remaining NO cases, add business_purpose field (separate from mission)
Phase 4: Re-validate all changed missions with LLM

Tracks before/after for every change. Saves snapshot for comparison.
"""
import json, re, pathlib, time, copy
from huggingface_hub import InferenceClient

# ── Config ──
REG_PATH = pathlib.Path("companies_registry.json")
SEC_DIR = pathlib.Path("MSGKG/data/data")
MODEL = "Qwen/Qwen2.5-72B-Instruct"
RESULTS_PATH = pathlib.Path("_mission_improvement_results.json")

client = InferenceClient(provider="novita")

with open(REG_PATH, encoding="utf-8") as f:
    data = json.load(f)

# ── Save before-snapshot ──
before_snapshot = {}
for cik, info in data.items():
    ov = info.get("overview", {})
    before_snapshot[cik] = {
        "name": info.get("name", ""),
        "mission": ov.get("mission", ""),
        "mission_quality": ov.get("mission_quality", 0),
        "llm_is_mission": ov.get("llm_is_mission", ""),
        "llm_actual_type": ov.get("llm_actual_type", ""),
        "llm_quality": ov.get("llm_quality", 0),
        "llm_confidence": ov.get("llm_confidence", 0),
    }


def get_item1_text(cik):
    """Get the full Item 1 (Business) section from the 10-K file."""
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

            # Try to find Item 1 boundaries
            item1_start = None
            for pat in [r"(?i)ITEM\s*1\.?\s*[\-—–]?\s*BUSINESS",
                        r"(?i)Item\s+1\.\s+Business"]:
                m = re.search(pat, text_norm)
                if m:
                    item1_start = m.start()
                    break

            item1_end = None
            if item1_start is not None:
                search_from = item1_start + 100
                for pat in [r"(?i)ITEM\s*1A\.?\s*[\-—–]?\s*RISK\s*FACTORS",
                            r"(?i)Item\s+1A\.\s+Risk\s+Factors",
                            r"(?i)ITEM\s*2\.?\s*[\-—–]?\s*PROPERTIES"]:
                    m = re.search(pat, text_norm[search_from:])
                    if m:
                        item1_end = search_from + m.start()
                        break

            if item1_start is not None:
                return text_norm[item1_start:item1_end] if item1_end else text_norm[item1_start:item1_start + 100000]
            else:
                return text_norm[:80000]
    return ""


def llm_extract_mission(company_name, sector, item1_text):
    """Ask LLM to extract the mission statement from Item 1 text."""
    # Send first 3000 chars (where missions typically are) + any mission-keyword regions
    chunks = []

    # First 3000 chars of Item 1
    chunks.append(("Beginning of Item 1", item1_text[:3000]))

    # Find regions with mission keywords
    text_lower = item1_text.lower()
    mission_keywords = ["our mission", "our purpose", "our vision", "mission is",
                        "purpose is", "we exist", "committed to", "we strive",
                        "we believe", "we aim", "core purpose", "we aspire"]
    seen_positions = set()
    for kw in mission_keywords:
        idx = text_lower.find(kw)
        if idx >= 0 and idx > 3000:
            # Avoid duplicate regions
            region_key = idx // 500
            if region_key not in seen_positions:
                seen_positions.add(region_key)
                start = max(0, idx - 200)
                end = min(len(item1_text), idx + 800)
                chunks.append((f"Region near '{kw}'", item1_text[start:end]))

    context = "\n\n---\n\n".join(f"[{label}]\n{text}" for label, text in chunks[:4])

    prompt = f"""You are a financial analyst extracting corporate mission statements from SEC 10-K filings.

Company: {company_name}
Sector: {sector}

Below are excerpts from the company's 10-K filing (Item 1 - Business section):

{context[:4000]}

Task: Extract the company's mission statement from this text.

Rules:
- A mission statement declares WHY the company exists and WHO it serves
- Look for phrases like "our mission is...", "our purpose is...", "we are committed to...", "our vision is..."
- Do NOT extract strategy, operations, HR policies, or financial goals
- Extract the COMPLETE sentence(s) — do not truncate
- If no explicit mission exists, extract the closest statement of PURPOSE (why the company exists)
- If nothing qualifies as mission or purpose, return "NONE"

Respond in EXACTLY this JSON format (no other text):
{{
  "found": true or false,
  "mission": "the extracted mission statement (complete sentence)",
  "source_type": "EXPLICIT_MISSION" or "VISION" or "PURPOSE" or "COMMITMENT" or "NONE",
  "extraction_context": "the 1-2 words before the mission in the source text",
  "confidence": 0 to 100
}}"""

    try:
        response = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(text[json_start:json_end])
        return {"error": "no_json", "raw": text[:200]}
    except Exception as e:
        return {"error": str(e)[:100]}


def llm_extract_business_purpose(company_name, sector, item1_text):
    """For companies with no mission, extract business purpose as a separate field."""
    prompt = f"""Company: {company_name}
Sector: {sector}

From this 10-K filing excerpt, write a one-sentence business purpose statement (what the company does and who it serves).
This is NOT a mission statement — it's a factual description of the company's core business.

Text:
{item1_text[:2500]}

Respond in EXACTLY this JSON format:
{{
  "business_purpose": "One sentence describing what the company does and who it serves",
  "confidence": 0 to 100
}}"""

    try:
        response = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(text[json_start:json_end])
        return {"error": "no_json"}
    except Exception as e:
        return {"error": str(e)[:100]}


def llm_validate_mission(company_name, sector, mission, context=""):
    """Quick LLM validation of a mission statement."""
    prompt = f"""Company: {company_name} | Sector: {sector}

Is this a genuine corporate mission statement?
"{mission}"

Context from filing: "{context[:500]}"

Respond in JSON:
{{
  "is_mission": "YES" or "NO" or "PARTIAL",
  "actual_type": "MISSION" or "VISION" or "STRATEGY" or "OPERATIONS" or "HR" or "FINANCIAL" or "MIXED",
  "mission_quality": 1 or 2 or 3 or 4 or 0,
  "quality_reason": "one sentence",
  "confidence": 0 to 100,
  "issues": []
}}

Quality: 1=Directly stated, 2=Requires inference, 3=Derived from context, 4=Vague, 0=Not a mission.
Be strict: missions declare WHY the company exists and WHO it serves."""

    try:
        response = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        text = response.choices[0].message.content.strip()
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(text[json_start:json_end])
        return {"error": "no_json"}
    except Exception as e:
        return {"error": str(e)[:100]}


# ═════════════════════════════════════════════════════════════════════════
# PHASE 1: Apply PARTIAL corrections
# ═════════════════════════════════════════════════════════════════════════
def phase1():
    print("=" * 70)
    print("PHASE 1: Apply 31 PARTIAL LLM Corrections")
    print("=" * 70)

    changes = []
    for cik, info in data.items():
        ov = info.get("overview", {})
        if ov.get("llm_is_mission") != "PARTIAL":
            continue

        corr = ov.get("llm_corrected_mission", "")
        if not corr or corr == "NONE" or len(corr) < 15:
            continue

        old_mission = ov.get("mission", "")
        name = info.get("name", "")[:35]

        ov["mission_before_improvement"] = old_mission
        ov["mission"] = corr
        ov["mission_improvement_phase"] = "phase1_partial_correction"

        changes.append({
            "cik": cik,
            "name": name,
            "old": old_mission[:60],
            "new": corr[:60],
            "phase": "phase1",
        })
        print(f"  [{len(changes)}] {name}")
        print(f"      OLD: {old_mission[:70]}...")
        print(f"      NEW: {corr[:70]}...")
        print()

    print(f"Phase 1 complete: {len(changes)} missions corrected\n")
    return changes


# ═════════════════════════════════════════════════════════════════════════
# PHASE 2: Re-extract for NO cases using full Item 1 + LLM
# ═════════════════════════════════════════════════════════════════════════
def phase2():
    print("=" * 70)
    print("PHASE 2: Re-Extract Missions for NO Cases via LLM")
    print("=" * 70)

    changes = []
    no_mission_ciks = []

    no_cases = [(cik, info) for cik, info in data.items()
                if info.get("overview", {}).get("llm_is_mission") == "NO"]

    print(f"Processing {len(no_cases)} NO cases...\n")

    for i, (cik, info) in enumerate(no_cases):
        ov = info.get("overview", {})
        name = info.get("name", "")[:35]
        sector = info.get("sector", "Unknown")
        old_mission = ov.get("mission", "")

        print(f"  [{i+1}/{len(no_cases)}] {name}...", end=" ", flush=True)

        item1 = get_item1_text(cik)
        if not item1 or len(item1) < 100:
            print("NO FILE")
            no_mission_ciks.append(cik)
            continue

        result = llm_extract_mission(name, sector, item1)
        time.sleep(1.5)

        if "error" in result:
            print(f"ERROR: {result['error'][:40]}")
            no_mission_ciks.append(cik)
            continue

        if result.get("found") and result.get("mission") and result["mission"] != "NONE":
            new_mission = result["mission"]
            source_type = result.get("source_type", "UNKNOWN")
            confidence = result.get("confidence", 0)

            ov["mission_before_improvement"] = old_mission
            ov["mission"] = new_mission
            ov["mission_improvement_phase"] = f"phase2_reextract_{source_type}"
            ov["reextract_confidence"] = confidence
            ov["reextract_source_type"] = source_type

            changes.append({
                "cik": cik, "name": name,
                "old": old_mission[:60], "new": new_mission[:60],
                "source_type": source_type, "confidence": confidence,
                "phase": "phase2",
            })
            print(f"FOUND ({source_type}, {confidence}%)")
            print(f"      OLD: {old_mission[:70]}")
            print(f"      NEW: {new_mission[:70]}")
            print()
        else:
            print("NONE FOUND")
            no_mission_ciks.append(cik)

    print(f"\nPhase 2 complete: {len(changes)} missions re-extracted")
    print(f"  Remaining NO: {len(no_mission_ciks)}\n")
    return changes, no_mission_ciks


# ═════════════════════════════════════════════════════════════════════════
# PHASE 3: Add business_purpose for remaining NO cases
# ═════════════════════════════════════════════════════════════════════════
def phase3(no_mission_ciks):
    print("=" * 70)
    print("PHASE 3: Extract Business Purpose for Remaining NO Cases")
    print("=" * 70)

    changes = []
    for i, cik in enumerate(no_mission_ciks):
        info = data[cik]
        ov = info.get("overview", {})
        name = info.get("name", "")[:35]
        sector = info.get("sector", "Unknown")

        print(f"  [{i+1}/{len(no_mission_ciks)}] {name}...", end=" ", flush=True)

        item1 = get_item1_text(cik)
        if not item1 or len(item1) < 100:
            print("NO FILE")
            ov["business_purpose"] = ""
            ov["mission_quality"] = 0
            continue

        result = llm_extract_business_purpose(name, sector, item1)
        time.sleep(1.5)

        if "error" in result:
            print(f"ERROR: {result['error'][:40]}")
            continue

        bp = result.get("business_purpose", "")
        if bp and len(bp) > 15:
            ov["business_purpose"] = bp
            ov["mission_improvement_phase"] = "phase3_business_purpose"
            ov["mission_quality"] = 0  # Keep M0 — this is NOT a mission

            changes.append({
                "cik": cik, "name": name,
                "business_purpose": bp[:80],
                "phase": "phase3",
            })
            print(f"OK")
            print(f"      PURPOSE: {bp[:80]}")
            print()
        else:
            print("NONE")

    print(f"\nPhase 3 complete: {len(changes)} business purposes extracted\n")
    return changes


# ═════════════════════════════════════════════════════════════════════════
# PHASE 4: Re-validate all changed missions
# ═════════════════════════════════════════════════════════════════════════
def phase4(phase1_changes, phase2_changes):
    print("=" * 70)
    print("PHASE 4: Re-Validate All Changed Missions")
    print("=" * 70)

    all_changed = phase1_changes + phase2_changes
    print(f"Re-validating {len(all_changed)} changed missions...\n")

    results = []
    for i, change in enumerate(all_changed):
        cik = change["cik"]
        info = data[cik]
        ov = info.get("overview", {})
        name = info.get("name", "")[:35]
        sector = info.get("sector", "Unknown")
        mission = ov.get("mission", "")

        print(f"  [{i+1}/{len(all_changed)}] {name}...", end=" ", flush=True)

        item1 = get_item1_text(cik)
        context = ""
        if item1:
            # Find mission in Item 1 for context
            idx = item1.lower().find(mission[:40].lower())
            if idx >= 0:
                context = item1[max(0, idx-200):idx+len(mission)+200]

        result = llm_validate_mission(name, sector, mission, context)
        time.sleep(1.5)

        if "error" in result:
            print(f"ERROR")
            continue

        is_m = result.get("is_mission", "?")
        mq = result.get("mission_quality", 0)
        conf = result.get("confidence", 0)
        atype = result.get("actual_type", "?")
        print(f"{is_m} | M{mq} | {conf}% | {atype}")

        # Update registry with new validation
        ov["llm_is_mission"] = result.get("is_mission", "")
        ov["llm_actual_type"] = result.get("actual_type", "")
        ov["llm_quality"] = result.get("mission_quality", 0)
        ov["llm_quality_reason"] = result.get("quality_reason", "")
        ov["llm_confidence"] = result.get("confidence", 0)
        ov["llm_issues"] = result.get("issues", [])

        if conf >= 60:
            ov["mission_quality"] = mq

        results.append({
            "cik": cik, "name": name,
            "is_mission": is_m, "quality": mq,
            "confidence": conf, "type": atype,
        })

    print(f"\nPhase 4 complete: {len(results)} re-validated")

    # Summary
    yes = sum(1 for r in results if r["is_mission"] == "YES")
    partial = sum(1 for r in results if r["is_mission"] == "PARTIAL")
    no = sum(1 for r in results if r["is_mission"] == "NO")
    print(f"  YES: {yes}, PARTIAL: {partial}, NO: {no}\n")
    return results


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("MISSION STATEMENT IMPROVEMENT PIPELINE")
    print("=" * 70 + "\n")

    # Phase 1
    p1_changes = phase1()

    # Phase 2
    p2_changes, no_ciks = phase2()

    # Phase 3
    p3_changes = phase3(no_ciks)

    # Phase 4
    p4_results = phase4(p1_changes, p2_changes)

    # ── Save registry ──
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Registry saved.\n")

    # ── Build after-snapshot and comparison ──
    after_snapshot = {}
    for cik, info in data.items():
        ov = info.get("overview", {})
        after_snapshot[cik] = {
            "name": info.get("name", ""),
            "mission": ov.get("mission", ""),
            "mission_quality": ov.get("mission_quality", 0),
            "llm_is_mission": ov.get("llm_is_mission", ""),
            "llm_actual_type": ov.get("llm_actual_type", ""),
            "llm_quality": ov.get("llm_quality", 0),
            "llm_confidence": ov.get("llm_confidence", 0),
            "business_purpose": ov.get("business_purpose", ""),
            "improvement_phase": ov.get("mission_improvement_phase", ""),
        }

    # ── Comparison report ──
    print("=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70 + "\n")

    # Distribution comparison
    before_mq = {}
    after_mq = {}
    before_llm = {"YES": 0, "NO": 0, "PARTIAL": 0}
    after_llm = {"YES": 0, "NO": 0, "PARTIAL": 0}

    for cik in data:
        b = before_snapshot[cik]
        a = after_snapshot[cik]
        bm = b.get("mission_quality", 0)
        am = a.get("mission_quality", 0)
        before_mq[bm] = before_mq.get(bm, 0) + 1
        after_mq[am] = after_mq.get(am, 0) + 1
        bl = b.get("llm_is_mission", "")
        al = a.get("llm_is_mission", "")
        if bl in before_llm:
            before_llm[bl] += 1
        if al in after_llm:
            after_llm[al] += 1

    print("Mission Quality Distribution:")
    print(f"  {'Code':<6} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'-'*32}")
    labels = {0: "M0", 1: "M1", 2: "M2", 3: "M3", 4: "M4"}
    for k in sorted(set(list(before_mq.keys()) + list(after_mq.keys()))):
        b = before_mq.get(k, 0)
        a = after_mq.get(k, 0)
        diff = a - b
        sign = "+" if diff > 0 else ""
        print(f"  {labels.get(k, k):<6} {b:>8} {a:>8} {sign}{diff:>7}")

    print(f"\nLLM Verdict Distribution:")
    print(f"  {'Verdict':<10} {'Before':>8} {'After':>8} {'Change':>8}")
    print(f"  {'-'*36}")
    for v in ["YES", "PARTIAL", "NO"]:
        b = before_llm.get(v, 0)
        a = after_llm.get(v, 0)
        diff = a - b
        sign = "+" if diff > 0 else ""
        print(f"  {v:<10} {b:>8} {a:>8} {sign}{diff:>7}")

    # List all changes
    print(f"\n{'='*70}")
    print("ALL CHANGES (Before -> After)")
    print(f"{'='*70}\n")

    changed_count = 0
    for cik in data:
        b = before_snapshot[cik]
        a = after_snapshot[cik]
        if b["mission"] != a["mission"] or a.get("business_purpose"):
            changed_count += 1
            phase = a.get("improvement_phase", "")
            print(f"  {changed_count}. {a['name'][:40]} [{phase}]")
            print(f"     BEFORE: {b['mission'][:80]}")
            if b["mission"] != a["mission"]:
                print(f"     AFTER:  {a['mission'][:80]}")
            if a.get("business_purpose"):
                print(f"     BIZ PURPOSE: {a['business_purpose'][:80]}")
            print(f"     LLM: {b['llm_is_mission']} -> {a['llm_is_mission']} | "
                  f"Quality: M{b['llm_quality']} → M{a['llm_quality']}")
            print()

    print(f"Total changes: {changed_count}")

    # Save full results
    full_results = {
        "before_snapshot": before_snapshot,
        "after_snapshot": after_snapshot,
        "phase1_changes": p1_changes,
        "phase2_changes": p2_changes,
        "phase3_changes": p3_changes,
        "phase4_results": p4_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
