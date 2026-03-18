"""Print before/after comparison from registry data."""
import json

with open("companies_registry.json", encoding="utf-8") as f:
    data = json.load(f)

print("=" * 70)
print("MISSION IMPROVEMENT: BEFORE vs AFTER")
print("=" * 70)

# Categorize all companies
changed = []
bp_added = []
unchanged = []
errors_need_retry = []

for cik, info in data.items():
    ov = info.get("overview", {})
    name = info["name"][:40]
    phase = ov.get("mission_improvement_phase", "")
    old_mission = ov.get("mission_before_improvement", "")
    new_mission = ov.get("mission", "")
    bp = ov.get("business_purpose", "")
    llm_is = ov.get("llm_is_mission", "")
    llm_type = ov.get("llm_actual_type", "")
    mq = ov.get("mission_quality", 0)

    if old_mission and old_mission != new_mission:
        changed.append((cik, name, old_mission, new_mission, phase, llm_is, llm_type, mq))
    elif bp:
        bp_added.append((cik, name, new_mission, bp, llm_is, llm_type, mq))
    elif llm_is == "NO" and not bp and not phase:
        errors_need_retry.append((cik, name, new_mission[:60], llm_type))

# Print mission changes
print(f"\n--- MISSION STATEMENTS IMPROVED ({len(changed)}) ---\n")
for i, (cik, name, old, new, phase, llm_is, llm_type, mq) in enumerate(changed, 1):
    print(f"{i:2d}. {name}")
    print(f"    Phase: {phase}")
    print(f"    BEFORE: {old[:85]}")
    print(f"    AFTER:  {new[:85]}")
    print(f"    LLM: {llm_is} | Type: {llm_type} | Quality: M{mq}")
    print()

# Print business purposes added
print(f"\n--- BUSINESS PURPOSE ADDED (no mission in filing) ({len(bp_added)}) ---\n")
for i, (cik, name, mission, bp, llm_is, llm_type, mq) in enumerate(bp_added, 1):
    print(f"{i:2d}. {name}")
    print(f"    Mission (kept): {mission[:75]}")
    print(f"    Biz Purpose:    {bp[:75]}")
    print()

# Print errors needing retry
if errors_need_retry:
    print(f"\n--- NEED RETRY (402 rate limit errors) ({len(errors_need_retry)}) ---\n")
    for cik, name, mission, ltype in errors_need_retry:
        print(f"  - {name} | {ltype} | {mission}")

# Summary distribution
print("\n" + "=" * 70)
print("DISTRIBUTION SUMMARY")
print("=" * 70)

mq_counts = {}
llm_counts = {"YES": 0, "PARTIAL": 0, "NO": 0}
type_counts = {}
has_bp = 0

for cik, info in data.items():
    ov = info.get("overview", {})
    mq = ov.get("mission_quality", 0)
    mq_counts[mq] = mq_counts.get(mq, 0) + 1
    li = ov.get("llm_is_mission", "")
    if li in llm_counts:
        llm_counts[li] += 1
    lt = ov.get("llm_actual_type", "")
    type_counts[lt] = type_counts.get(lt, 0) + 1
    if ov.get("business_purpose"):
        has_bp += 1

labels = {0: "M0-Absent", 1: "M1-Direct", 2: "M2-Inference", 3: "M3-Context", 4: "M4-Vague"}
print("\nMission Quality:")
for k in sorted(mq_counts):
    print(f"  {labels.get(k, k):15s}: {mq_counts[k]}")

print(f"\nLLM Verdict:")
for v, c in llm_counts.items():
    print(f"  {v:10s}: {c}")

print(f"\nContent Type:")
for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t:15s}: {c}")

print(f"\nBusiness purpose (separate field): {has_bp} companies")
print(f"Total companies: {len(data)}")
