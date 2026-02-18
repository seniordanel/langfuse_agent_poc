import os
import uuid
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from src.graph import app
from src.evals import run_eval_suite


# ─── Research Topics for Multi-Session Demo ─────────────────
TOPICS = [
    {
        "task": "Future of Edge Computing in 2026",
        "tags": ["edge-computing", "infrastructure", "6G"],
    },
    {
        "task": "Quantum Machine Learning: Promise vs Reality in 2026",
        "tags": ["quantum", "machine-learning", "research"],
    },
    {
        "task": "Global AI Regulation Landscape in 2026",
        "tags": ["ai-regulation", "policy", "governance"],
    },
]


def run_single_topic(langfuse: Langfuse, session_id: str, topic: dict, run_index: int):
    """Runs the full 9-agent pipeline for a single topic."""
    task = topic["task"]
    tags = ["stress-test", "v3-enhanced"] + topic["tags"]

    print(f"\n{'#'*70}")
    print(f"# RUN {run_index + 1}/{len(TOPICS)}: {task}")
    print(f"{'#'*70}")

    # Create a trace ID for this run
    trace_id = langfuse.create_trace_id()
    print(f"  Trace ID: {trace_id}")
    print(f"  Session:  {session_id}")
    print(f"  Tags:     {tags}")

    # Create the Langfuse callback handler linked to our trace
    langfuse_handler = CallbackHandler(
        trace_context={"trace_id": trace_id},
        update_trace=True,
    )

    # Initial state
    initial_state = {
        "task": task,
        "topic_label": task,
        "research_data": [],
        "analysis": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "compliance_notes": "",
        "compliance_revision_count": 0,
        "seo_keywords": [],
        "executive_summary": "",
        "translated_summaries": {},
        "final_output": "",
        "iteration_log": [],
        "messages": [HumanMessage(content=task)],
        # ── New fields for enhanced pipeline ──
        "enrichment_data": [],
        "quality_score": 0.0,
        "quality_revision_count": 0,
        "sentiment_scores": {},
        "readability_grade": 0.0,
    }

    start_time = time.time()

    try:
        result = app.invoke(
            initial_state,
            config={"callbacks": [langfuse_handler]},
        )

        end_time = time.time()
        latency = end_time - start_time

        # Extract results
        final_draft = result.get("draft", "")
        exec_summary = result.get("executive_summary", "")
        translations = result.get("translated_summaries", {})
        seo_kw = result.get("seo_keywords", [])
        iteration_log = result.get("iteration_log", [])

        print(f"\n  --- Results ---")
        print(f"  Draft:        {len(final_draft)} chars")
        print(f"  Latency:      {latency:.1f}s")
        print(f"  SEO Keywords: {seo_kw}")
        print(f"  Iterations:   {' → '.join(iteration_log)}")
        print(f"  Exec Summary: {exec_summary[:150]}...")

        if translations:
            for lang, text in translations.items():
                print(f"  Translation [{lang}]: {str(text)[:80]}...")

        # Run 8-score evaluation suite
        research_str = "\n".join(result.get("research_data", []))
        run_eval_suite(
            trace_id=trace_id,
            output_text=final_draft,
            research_data=research_str,
            latency=latency,
            cost=0.03 * (run_index + 1),  # simulated varying cost per run
        )

        return {
            "trace_id": trace_id,
            "task": task,
            "latency": latency,
            "draft_length": len(final_draft),
            "iterations": len(iteration_log),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n  ERROR in run {run_index + 1}: {e}")
        return {"trace_id": trace_id, "task": task, "error": str(e)}


def main():
    print("=" * 70)
    print("  MULTI-AGENT RESEARCH SYSTEM — Enhanced Langfuse Demo")
    print("  11 Agents | 12 Tools | 3 Topics | 3 Feedback Loops | 8 Evaluations per Trace")
    print("=" * 70)

    langfuse = Langfuse(timeout=120)

    # Single session groups all 3 runs together in Langfuse
    session_id = str(uuid.uuid4())
    print(f"\nSession ID: {session_id}")
    print(f"Langfuse:   {os.getenv('LANGFUSE_BASE_URL', 'http://localhost:3000')}")

    results = []
    total_start = time.time()

    for i, topic in enumerate(TOPICS):
        result = run_single_topic(langfuse, session_id, topic, i)
        results.append(result)

    total_time = time.time() - total_start

    # ─── Final Summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  ALL RUNS COMPLETE — {len(TOPICS)} topics in {total_time:.1f}s")
    print(f"{'='*70}")

    for r in results:
        status = "✅" if "error" not in r else "❌"
        print(f"  {status} {r['task']}")
        if "error" not in r:
            print(f"     Trace: {r['trace_id']} | {r['latency']:.1f}s | {r['draft_length']} chars | {r['iterations']} steps")
        else:
            print(f"     Error: {r['error']}")

    print(f"\n  Session: {session_id}")
    print(f"  Total scores submitted: {len(results) * 8}")

    # Final flush
    langfuse.flush()
    print(f"\n  ✅ All data flushed to Langfuse. Check your dashboard!")
    print(f"  URL: {os.getenv('LANGFUSE_BASE_URL', 'http://localhost:3000')}")


if __name__ == "__main__":
    main()
