"""
LLM-based mission evaluation using Qwen2.5-72B-Instruct.

Pre-computes nuanced scores for all 8 quality dimensions with reasoning.
Stores results in companies_registry.json under overview.llm_eval.

Run: python llm_evaluate_missions.py
"""

import json, time, pathlib
from huggingface_hub import InferenceClient

REG_PATH = pathlib.Path("companies_registry.json")
MODEL = "Qwen/Qwen2.5-72B-Instruct"
client = InferenceClient(provider="novita")

RUBRIC = """Score each dimension 0-4 using this rubric:

1. Clarity of Purpose (0-4): Does the statement clearly articulate WHY the organization exists?
   4=Specific, concrete purpose with clear domain. 3=Good purpose but slightly generic. 2=Purpose present but vague. 1=Very unclear. 0=No purpose.

2. Stakeholder Focus (0-4): Does it identify WHO the company serves or impacts?
   4=3+ stakeholder groups named. 3=2 groups. 2=1 group. 1=Implied but not stated. 0=None.

3. Value Proposition (0-4): Does it describe WHAT unique value the company delivers?
   4=Clear, differentiated value with delivery mechanism. 3=Good value statement. 2=Generic value. 1=Weak/implied. 0=None.

4. Actionability (0-4): Is the language concrete and action-oriented vs abstract?
   4=Specific actions, measurable outcomes. 3=Good action verbs, clear direction. 2=Some action language. 1=Mostly abstract. 0=No actionable content.

5. Innovation Signal (0-4): Does it signal forward-looking, differentiating ambition?
   4=Strong innovation/transformation language with specifics. 3=Good forward-looking signals. 2=Some innovation language. 1=Generic "leading" claims. 0=None.

6. Semantic Completeness (0-4): Does it cover Purpose(Why), Activity(What), Audience(Who), Method(How), Domain(Where)?
   4=All 5 present. 3=4 of 5. 2=3 of 5. 1=1-2 of 5. 0=None.

7. Internal Consistency (0-4): Is the message coherent without contradictions or buzzword padding?
   4=Clean, focused, no contradictions. 3=Mostly clean. 2=Some buzzwords/filler. 1=Contradictory or heavily padded. 0=Incoherent.

8. Writing Quality (0-4): Is it well-written, concise, and professional?
   4=Excellent prose, concise (10-30 words), active voice. 3=Good writing. 2=Adequate but wordy or passive. 1=Poor structure. 0=Unreadable/missing.
"""

PROMPT_TEMPLATE = """Company: {name}
Sector: {sector}

Mission statement to evaluate:
"{mission}"

{rubric}

Evaluate this mission statement. For each dimension, provide:
- A score (0-4)
- A 1-sentence justification

Also provide:
- overall_assessment: 2-sentence summary of the mission's strengths and weaknesses
- benchmark_comparison: How does this compare to best-in-class mission statements in the {sector} sector?

Return ONLY valid JSON:
{{
  "dimensions": {{
    "Clarity of Purpose": {{"score": N, "reason": "..."}},
    "Stakeholder Focus": {{"score": N, "reason": "..."}},
    "Value Proposition": {{"score": N, "reason": "..."}},
    "Actionability": {{"score": N, "reason": "..."}},
    "Innovation Signal": {{"score": N, "reason": "..."}},
    "Semantic Completeness": {{"score": N, "reason": "..."}},
    "Internal Consistency": {{"score": N, "reason": "..."}},
    "Writing Quality": {{"score": N, "reason": "..."}}
  }},
  "overall_assessment": "...",
  "benchmark_comparison": "..."
}}"""


def evaluate_company(cik, info):
    """Evaluate a single company's mission using LLM."""
    ov = info.get("overview", {})
    name = info["name"][:50]
    sector = info.get("sector", "Unknown")
    mission = ov.get("mission", "")

    if not mission or len(mission.strip()) < 10:
        return None

    prompt = PROMPT_TEMPLATE.format(
        name=name, sector=sector, mission=mission, rubric=RUBRIC
    )

    try:
        resp = client.chat_completion(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in corporate strategy and mission statement analysis. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(text[start:end])
            return result
    except Exception as e:
        print(f"    ERROR: {str(e)[:80]}")
    return None


def main():
    with open(REG_PATH, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    done = 0
    skipped = 0
    errors = 0

    print(f"LLM Mission Evaluation: {total} companies")
    print(f"Model: {MODEL}")
    print("=" * 60)

    for cik, info in data.items():
        ov = info.get("overview", {})
        name = info["name"][:40]

        # Skip if already evaluated
        if ov.get("llm_eval") and ov["llm_eval"].get("dimensions"):
            done += 1
            print(f"  [{done}/{total}] {name} ... CACHED")
            continue

        mission = ov.get("mission", "")
        if not mission or len(mission.strip()) < 10:
            skipped += 1
            print(f"  [{done + skipped}/{total}] {name} ... NO MISSION")
            continue

        print(f"  [{done + skipped + 1}/{total}] {name} ...", end=" ", flush=True)

        # Retry with backoff on 402 errors
        result = None
        for attempt in range(3):
            result = evaluate_company(cik, info)
            if result is not None:
                break
            wait = 30 * (attempt + 1)
            print(f"    Retry in {wait}s...", end=" ", flush=True)
            time.sleep(wait)

        if result and "dimensions" in result:
            ov["llm_eval"] = result
            # Compute LLM overall score
            scores = [d["score"] for d in result["dimensions"].values()
                      if isinstance(d, dict) and "score" in d]
            if scores:
                ov["llm_eval"]["overall_score"] = round(sum(scores) / len(scores), 2)
            done += 1
            avg = ov["llm_eval"].get("overall_score", 0)
            print(f"OK (avg={avg:.1f})")
        else:
            errors += 1
            print("FAILED")

        time.sleep(12)  # Rate limit (increased for free tier)

    # Save
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Done: {done} | Skipped: {skipped} | Errors: {errors}")
    print(f"Registry saved to {REG_PATH}")


if __name__ == "__main__":
    main()
