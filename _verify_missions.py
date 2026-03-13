"""Automated mission statement verification pipeline."""
import json, re, pathlib

reg_path = pathlib.Path("companies_registry.json")
with open(reg_path, encoding="utf-8") as f:
    data = json.load(f)

sec_dir = pathlib.Path("MSGKG/data/data")

# M1 context keywords (explicit mission language)
M1_KEYWORDS = ["our mission", "our purpose", "our vision", "mission statement", "corporate mission",
               "the mission of", "core purpose", "company mission", "mission is to"]
# M2 context keywords (implicit commitment language)
M2_KEYWORDS = ["committed to", "dedicated to", "we strive", "we believe", "we aim",
               "our goal", "we seek", "we aspire", "focused on"]
# Business description (M3)
M3_KEYWORDS = ["we provide", "we offer", "we deliver", "we operate", "we serve",
               "we develop", "we help", "our strategy", "core business"]

results = []
reclassified = 0
total_verified = 0

for cik, info in data.items():
    ov = info.get("overview", {})
    mission = ov.get("mission", "")
    old_mq = ov.get("mission_quality", 0)
    name = info.get("name", "?")

    if not mission or len(mission) < 10:
        ov["mission_quality"] = 0
        ov["verification_status"] = "NO_MISSION"
        ov["verification_match"] = "none"
        ov["verification_context"] = ""
        results.append({"name": name, "old": old_mq, "new": 0, "v": "NO_MISSION", "match": "none", "changed": old_mq != 0})
        continue

    # Find 10-K file
    cik_stripped = cik.lstrip("0")
    txt_file = None
    for f in sec_dir.glob("cleaned_10-K_*.txt"):
        acc = f.stem.replace("cleaned_10-K_", "")
        file_cik = acc.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            txt_file = f
            break

    if not txt_file:
        ov["verification_status"] = "NO_FILE"
        ov["verification_match"] = "none"
        ov["verification_context"] = ""
        results.append({"name": name, "old": old_mq, "new": old_mq, "v": "NO_FILE", "match": "none", "changed": False})
        continue

    text = txt_file.read_text(encoding="utf-8", errors="ignore")
    hdr = text.find("</SEC-HEADER>")
    if hdr > 0:
        text = text[hdr + len("</SEC-HEADER>"):]
    text_norm = re.sub(r"\s+", " ", text)
    text_lower = text_norm.lower()

    mission_lower = re.sub(r"\s+", " ", mission).strip().lower()

    # Step 1: Back-verification
    idx = text_lower.find(mission_lower[:50])
    match_type = "none"
    context_before = ""

    if idx >= 0:
        match_type = "exact"
        context_before = text_norm[max(0, idx - 300):idx].strip()
    else:
        idx = text_lower.find(mission_lower[:30])
        if idx >= 0:
            match_type = "partial"
            context_before = text_norm[max(0, idx - 300):idx].strip()
        else:
            mission_words = set(re.findall(r"\b[a-z]{4,}\b", mission_lower)) - {
                "the", "and", "for", "that", "with", "from", "are", "our", "has", "have",
                "been", "will", "their", "this", "not", "but", "which", "into", "also"
            }
            best_score = 0
            best_pos = 0
            for i in range(0, min(len(text_lower), 500000), 300):
                chunk = text_lower[i:i + 600]
                score = sum(1 for w in mission_words if w in chunk)
                if score > best_score:
                    best_score = score
                    best_pos = i
            if best_score >= 3:
                match_type = "keyword"
                context_before = text_norm[max(0, best_pos - 300):best_pos].strip()

    # Step 2: Context window validation
    context_lower = context_before.lower() if context_before else ""
    if idx is not None and idx >= 0:
        context_after = text_norm[idx:idx + len(mission) + 200].lower()
        full_context = context_lower + " " + context_after
    else:
        full_context = context_lower

    m1_hits = sum(1 for kw in M1_KEYWORDS if kw in full_context)
    m2_hits = sum(1 for kw in M2_KEYWORDS if kw in full_context)
    m3_hits = sum(1 for kw in M3_KEYWORDS if kw in full_context)

    # Step 3: Auto-reclassify
    if match_type == "exact" and m1_hits >= 1:
        new_mq = 1
        confidence = min(0.95, 0.70 + m1_hits * 0.08)
        verification = "VERIFIED_M1"
    elif match_type in ("exact", "partial") and m2_hits >= 1:
        new_mq = 2
        confidence = min(0.85, 0.55 + m2_hits * 0.08)
        verification = "VERIFIED_M2"
    elif match_type in ("exact", "partial") and m1_hits == 0 and m2_hits == 0:
        new_mq = 2
        confidence = 0.50
        verification = "TEXT_MATCH_NO_CONTEXT"
    elif match_type == "keyword" and (m1_hits >= 1 or m2_hits >= 1):
        new_mq = 3
        confidence = 0.40
        verification = "KEYWORD_WITH_CONTEXT"
    elif match_type == "keyword":
        new_mq = 3
        confidence = 0.30
        verification = "KEYWORD_ONLY"
    elif match_type == "none" and len(mission) > 30:
        new_mq = 4
        confidence = 0.15
        verification = "NO_MATCH"
    else:
        new_mq = 4
        confidence = 0.10
        verification = "LOW_CONFIDENCE"

    changed = old_mq != new_mq
    if changed:
        reclassified += 1

    ov["mission_quality"] = new_mq
    ov["mission_confidence"] = round(confidence, 2)
    ov["verification_status"] = verification
    ov["verification_match"] = match_type
    ov["verification_context"] = context_before[:200] if context_before else ""

    results.append({"name": name, "old": old_mq, "new": new_mq, "v": verification, "match": match_type, "changed": changed})
    total_verified += 1

# Save
with open(reg_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Report
print(f"Verified: {total_verified}/91")
print(f"Reclassified: {reclassified}")
print()

mq_counts = {}
for r in results:
    mq = r["new"]
    mq_counts[mq] = mq_counts.get(mq, 0) + 1
labels = {0: "M0-Absent", 1: "M1-Direct", 2: "M2-Inference", 3: "M3-Context", 4: "M4-Vague"}
print("New Distribution:")
for k in sorted(mq_counts):
    print(f"  {labels.get(k, k)}: {mq_counts[k]}")

print()
print("Verification Status:")
v_counts = {}
for r in results:
    v = r["v"]
    v_counts[v] = v_counts.get(v, 0) + 1
for v, c in sorted(v_counts.items(), key=lambda x: -x[1]):
    print(f"  {v}: {c}")

print()
print("Reclassified:")
for r in results:
    if r["changed"]:
        print(f"  {r['name'][:35]}: M{r['old']} -> M{r['new']} ({r['v']})")
