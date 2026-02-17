import re
from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse
from langchain_openai import ChatOpenAI

# Initialize Langfuse and LLM
langfuse = Langfuse(timeout=120)
llm_judge = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)


def _extract_score(text: str) -> float:
    """Extract first integer from LLM response, normalize to 0-1."""
    match = re.search(r"\d+", text.strip())
    if match:
        return min(float(match.group()) / 10.0, 1.0)
    return 0.5


# ════════════════════════════════════════════════════════════════
# DETERMINISTIC EVALUATIONS
# ════════════════════════════════════════════════════════════════

def eval_format_compliance(text: str) -> dict:
    """Checks mandatory sections and forbidden phrases."""
    issues = []

    required = [
        ("Executive Summary", ["## Executive Summary", "## executive summary"]),
        ("Introduction", ["## Introduction", "# Introduction"]),
        ("Conclusion", ["## Conclusion", "# Conclusion", "## Summary", "## Final Thoughts"]),
    ]
    for name, variations in required:
        if not any(v.lower() in text.lower() for v in variations):
            issues.append(f"Missing '{name}' section")

    forbidden = ["As an AI", "language model", "I cannot", "I'm sorry"]
    for phrase in forbidden:
        if phrase.lower() in text.lower():
            issues.append(f"Contains forbidden: '{phrase}'")

    score = 1.0 if not issues else 0.0
    return {"score": score, "reason": "; ".join(issues) if issues else "All checks passed"}


def eval_word_count(text: str) -> dict:
    """Checks minimum word count (500 words)."""
    count = len(text.split())
    if count >= 500:
        return {"score": 1.0, "reason": f"Word count: {count} (≥500)"}
    return {"score": round(count / 500.0, 2), "reason": f"Word count: {count} (<500 minimum)"}


def eval_has_references(text: str) -> dict:
    """Checks for presence of URLs or citation markers."""
    url_pattern = r"https?://\S+"
    urls = re.findall(url_pattern, text)
    citation_markers = re.findall(r"\[\d+\]", text)

    total = len(urls) + len(citation_markers)
    if total >= 2:
        return {"score": 1.0, "reason": f"Found {len(urls)} URLs, {len(citation_markers)} citations"}
    elif total >= 1:
        return {"score": 0.5, "reason": f"Only {total} reference(s) found"}
    return {"score": 0.0, "reason": "No references or citations found"}


# ════════════════════════════════════════════════════════════════
# LLM-AS-JUDGE EVALUATIONS
# ════════════════════════════════════════════════════════════════

def eval_analytical_rigor(text: str) -> dict:
    """LLM judge: depth of analysis, data usage, logical flow."""
    prompt = f"""Rate this report for ANALYTICAL RIGOR on a scale of 1-10.

Criteria:
- Does it go beyond surface-level observations?
- Are specific data points cited (numbers, percentages, dates)?
- Is there a logical argument structure?
- Are counter-arguments or risks addressed?

Report (first 3000 chars):
{text[:3000]}

Output ONLY a single integer 1-10."""
    try:
        resp = llm_judge.invoke(prompt)
        score = _extract_score(resp.content)
        return {"score": score, "reason": f"Analytical rigor: {resp.content.strip()}/10"}
    except Exception as e:
        return {"score": 0.0, "reason": f"Failed: {e}"}


def eval_readability(text: str) -> dict:
    """LLM judge: clarity, flow, accessibility."""
    prompt = f"""Rate this report for READABILITY on a scale of 1-10.

Criteria:
- Is the language clear and jargon-free (or jargon explained)?
- Do paragraphs flow logically from one to the next?
- Is the report scannable (good headings, short paragraphs)?
- Would a non-expert understand the key points?

Report (first 3000 chars):
{text[:3000]}

Output ONLY a single integer 1-10."""
    try:
        resp = llm_judge.invoke(prompt)
        score = _extract_score(resp.content)
        return {"score": score, "reason": f"Readability: {resp.content.strip()}/10"}
    except Exception as e:
        return {"score": 0.0, "reason": f"Failed: {e}"}


def eval_factual_consistency(text: str, research_data: str) -> dict:
    """LLM judge: how well the draft aligns with source research."""
    prompt = f"""Rate the FACTUAL CONSISTENCY of this report vs the source research on a scale of 1-10.

Source Research:
{research_data[:2000]}

Report:
{text[:2000]}

Criteria:
- Do claims in the report match the research data?
- Are there hallucinated facts not in the source?
- Are numbers and statistics accurately represented?

Output ONLY a single integer 1-10."""
    try:
        resp = llm_judge.invoke(prompt)
        score = _extract_score(resp.content)
        return {"score": score, "reason": f"Factual consistency: {resp.content.strip()}/10"}
    except Exception as e:
        return {"score": 0.0, "reason": f"Failed: {e}"}


# ════════════════════════════════════════════════════════════════
# PERFORMANCE EVALUATIONS
# ════════════════════════════════════════════════════════════════

def eval_latency(latency: float, threshold: float = 90.0) -> dict:
    ok = latency <= threshold
    return {
        "score": 1.0 if ok else 0.0,
        "reason": f"Latency: {latency:.1f}s {'≤' if ok else '>'} {threshold}s",
    }


def eval_cost(cost: float, threshold: float = 0.10) -> dict:
    ok = cost <= threshold
    return {
        "score": 1.0 if ok else 0.0,
        "reason": f"Cost: ${cost:.4f} {'≤' if ok else '>'} ${threshold}",
    }


# ════════════════════════════════════════════════════════════════
# MASTER EVAL RUNNER
# ════════════════════════════════════════════════════════════════

def run_eval_suite(
    trace_id: str,
    output_text: str,
    research_data: str,
    latency: float,
    cost: float,
):
    """Runs all 8 evaluations and posts scores to Langfuse."""
    print(f"\n{'='*60}")
    print(f"Running 8 evaluations for Trace: {trace_id}")
    print(f"{'='*60}")

    evals = [
        ("format_compliance", eval_format_compliance(output_text)),
        ("word_count_check", eval_word_count(output_text)),
        ("has_references", eval_has_references(output_text)),
        ("analytical_rigor", eval_analytical_rigor(output_text)),
        ("readability", eval_readability(output_text)),
        ("factual_consistency", eval_factual_consistency(output_text, research_data)),
        ("latency_check", eval_latency(latency)),
        ("cost_check", eval_cost(cost)),
    ]

    for name, result in evals:
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=result["score"],
            comment=result["reason"],
        )
        status = "✅" if result["score"] >= 0.7 else "⚠️" if result["score"] >= 0.4 else "❌"
        print(f"  {status} {name}: {result['score']:.2f}  ({result['reason']})")

    langfuse.flush()
    print(f"{'='*60}")
    print(f"All 8 scores submitted to Langfuse.\n")
