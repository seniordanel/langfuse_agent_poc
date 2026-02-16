"""
Langfuse Dataset Experiment Runner

This script demonstrates Langfuse's Dataset & Experiment features:
1. Creates a dataset in Langfuse with research topics.
2. Runs each dataset item through the 9-agent pipeline.
3. Evaluates each run with 8 scores.
4. Links runs to dataset items as experiment results.

Usage:
    .venv/bin/python run_dataset_experiment.py
"""

import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langfuse.langchain import CallbackHandler
from langfuse import Langfuse
from src.graph import app
from src.evals import run_eval_suite


DATASET_NAME = "research-topics-benchmark-v1"
EXPERIMENT_NAME = "9-agent-pipeline-v3"


def create_or_get_dataset(langfuse: Langfuse) -> None:
    """Create the dataset in Langfuse and populate it from eval_dataset.json."""
    print(f"\n📦 Creating dataset: {DATASET_NAME}")

    with open("eval_dataset.json", "r") as f:
        items = json.load(f)

    try:
        langfuse.create_dataset(
            name=DATASET_NAME,
            description="Research topics for benchmarking the 9-agent pipeline",
        )
        print(f"  Created new dataset: {DATASET_NAME}")
    except Exception:
        print(f"  Dataset '{DATASET_NAME}' already exists, reusing it")

    for i, item in enumerate(items):
        langfuse.create_dataset_item(
            dataset_name=DATASET_NAME,
            input={"topic": item["topic"]},
            expected_output={
                "expected_sections": item["expected_sections"],
                "min_word_count": item["min_word_count"],
                "expected_keywords": item["expected_keywords"],
            },
            metadata={"quality_threshold": item["quality_threshold"], "index": i},
        )
        print(f"  Added item {i+1}: {item['topic']}")

    langfuse.flush()
    print(f"  ✅ Dataset populated with {len(items)} items\n")


def run_experiment(langfuse: Langfuse):
    """Run each dataset item through the pipeline and link to experiment."""
    print(f"🧪 Running experiment: {EXPERIMENT_NAME}")

    dataset = langfuse.get_dataset(DATASET_NAME)
    results = []

    for i, item in enumerate(dataset.items):
        topic = item.input["topic"]
        expected = item.expected_output

        print(f"\n{'─'*60}")
        print(f"  Item {i+1}/{len(dataset.items)}: {topic}")
        print(f"{'─'*60}")

        # Create trace for this run
        trace_id = langfuse.create_trace_id()

        langfuse_handler = CallbackHandler(
            trace_context={"trace_id": trace_id},
            update_trace=True,
        )

        initial_state = {
            "task": topic,
            "topic_label": topic,
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
            "messages": [HumanMessage(content=topic)],
        }

        start = time.time()

        try:
            result = app.invoke(
                initial_state,
                config={"callbacks": [langfuse_handler]},
            )

            latency = time.time() - start
            draft = result.get("draft", "")
            research = "\n".join(result.get("research_data", []))

            # Run evaluations
            run_eval_suite(
                trace_id=trace_id,
                output_text=draft,
                research_data=research,
                latency=latency,
                cost=0.03,
            )

            # Link this run to the dataset item
            item.link(
                trace_id=trace_id,
                run_name=EXPERIMENT_NAME,
            )

            # Additional dataset-specific checks
            check_results = check_expected_output(draft, expected)

            results.append({
                "topic": topic,
                "trace_id": trace_id,
                "latency": latency,
                "draft_length": len(draft),
                "checks": check_results,
            })

            print(f"  ✅ Completed in {latency:.1f}s ({len(draft)} chars)")
            for check_name, passed in check_results.items():
                print(f"     {'✅' if passed else '❌'} {check_name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"topic": topic, "trace_id": trace_id, "error": str(e)})
            print(f"  ❌ Failed: {e}")

    langfuse.flush()
    return results


def check_expected_output(draft: str, expected: dict) -> dict:
    """Verify draft against dataset item's expected output."""
    checks = {}

    # Section presence check
    for section in expected.get("expected_sections", []):
        checks[f"has_{section.lower().replace(' ', '_')}"] = (
            section.lower() in draft.lower()
        )

    # Word count check
    wc = len(draft.split())
    checks["meets_word_count"] = wc >= expected.get("min_word_count", 500)

    # Keyword presence
    keywords = expected.get("expected_keywords", [])
    found = sum(1 for kw in keywords if kw.lower() in draft.lower())
    checks[f"keywords_{found}/{len(keywords)}"] = (found >= len(keywords) // 2)

    return checks


def main():
    print("=" * 70)
    print("  LANGFUSE DATASET EXPERIMENT RUNNER")
    print("  Pipeline: 9-Agent Research System")
    print("=" * 70)

    langfuse = Langfuse(timeout=120)

    # Step 1: Create/populate dataset
    create_or_get_dataset(langfuse)

    # Step 2: Run experiment
    results = run_experiment(langfuse)

    # Step 3: Summary
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPLETE: {EXPERIMENT_NAME}")
    print(f"{'='*70}")

    passed = sum(1 for r in results if "error" not in r)
    print(f"\n  Results: {passed}/{len(results)} runs succeeded")
    for r in results:
        if "error" not in r:
            checks_passed = sum(1 for v in r["checks"].values() if v)
            total_checks = len(r["checks"])
            print(f"  ✅ {r['topic']}: {r['latency']:.1f}s, {checks_passed}/{total_checks} checks passed")
        else:
            print(f"  ❌ {r['topic']}: {r['error']}")

    print(f"\n  Dashboard: {os.getenv('LANGFUSE_BASE_URL', 'http://localhost:3000')}")
    print(f"  Dataset:   {DATASET_NAME}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    langfuse.flush()


if __name__ == "__main__":
    main()
