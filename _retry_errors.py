"""Retry the 7 companies that hit 402 rate limits."""
import json, re, time, pathlib
from huggingface_hub import InferenceClient

REG_PATH = pathlib.Path("companies_registry.json")
SEC_DIR = pathlib.Path("MSGKG/data/data")
MODEL = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(provider="novita")

with open(REG_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Find companies that still need processing
retry = []
for cik, info in data.items():
    ov = info.get("overview", {})
    phase = ov.get("mission_improvement_phase", "")
    llm_is = ov.get("llm_is_mission", "")
    bp = ov.get("business_purpose", "")
    if llm_is == "NO" and not bp and not phase:
        retry.append(cik)

print(f"Retrying {len(retry)} companies...\n")


def get_item1_text(cik):
    cik_stripped = cik.lstrip("0")
    for f in SEC_DIR.glob("cleaned_10-K_*.txt"):
        acc = f.stem.replace("cleaned_10-K_", "")
        file_cik = acc.split("-")[0].lstrip("0")
        if file_cik == cik_stripped:
            text = f.read_text(encoding="utf-8", errors="ignore")
            hdr = text.find("</SEC-HEADER>")
            if hdr > 0:
                text = text[hdr + len("</SEC-HEADER>"):]
            return re.sub(r"\s+", " ", text)[:80000]
    return ""


for cik in retry:
    info = data[cik]
    ov = info.get("overview", {})
    name = info["name"][:40]
    sector = info.get("sector", "Unknown")
    print(f"  {name}...", end=" ", flush=True)

    item1 = get_item1_text(cik)
    if not item1:
        print("NO FILE")
        continue

    # Try mission extraction first
    chunks = [("Beginning", item1[:3000])]
    text_lower = item1.lower()
    for kw in ["our mission", "our purpose", "our vision", "committed to", "we believe"]:
        idx = text_lower.find(kw)
        if idx > 3000:
            chunks.append((kw, item1[max(0, idx - 200):idx + 800]))

    context = "\n---\n".join(f"[{l}]\n{t}" for l, t in chunks[:4])

    try:
        resp = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Respond with valid JSON only."},
                {"role": "user", "content": f"Company: {name} | Sector: {sector}\n\nExtract the mission statement (WHY the company exists, WHO it serves) from this 10-K. If none, return NONE.\n\n{context[:4000]}\n\nJSON: {{\"found\": true/false, \"mission\": \"...\", \"source_type\": \"EXPLICIT_MISSION/VISION/PURPOSE/COMMITMENT/NONE\", \"confidence\": 0-100}}"},
            ],
            max_tokens=400,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        result = json.loads(text[text.find("{"):text.rfind("}") + 1])

        if result.get("found") and result.get("mission", "NONE") != "NONE":
            old = ov.get("mission", "")
            ov["mission_before_improvement"] = old
            ov["mission"] = result["mission"]
            ov["mission_improvement_phase"] = f"phase2_reextract_{result.get('source_type', 'UNKNOWN')}"
            print(f"MISSION: {result['mission'][:65]}")
        else:
            # Business purpose fallback
            time.sleep(3)
            resp2 = client.chat_completion(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Respond with valid JSON only."},
                    {"role": "user", "content": f"Company: {name} | Sector: {sector}\n\nWrite ONE sentence: what the company does and who it serves.\n\n{item1[:2500]}\n\nJSON: {{\"business_purpose\": \"...\"}}"},
                ],
                max_tokens=200,
                temperature=0.1,
            )
            text2 = resp2.choices[0].message.content.strip()
            r2 = json.loads(text2[text2.find("{"):text2.rfind("}") + 1])
            bp = r2.get("business_purpose", "")
            if bp:
                ov["business_purpose"] = bp
                ov["mission_improvement_phase"] = "phase3_business_purpose"
                print(f"BIZ PURPOSE: {bp[:65]}")
            else:
                print("NONE")
    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")

    time.sleep(3)

with open(REG_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("\nRegistry updated.")
