#!/usr/bin/env python3
"""Run the RAGAS-style eval over eval/golden_set.json and write
eval/report.md + eval/results.json. Exits non-zero (CI gate failure) if
metrics fall below the thresholds in ragcite.eval.runner.DEFAULT_GATE_THRESHOLDS.

Usage:
    python scripts/run_eval.py                 # uses configured LLM_PROVIDER (default: mock)
    LLM_PROVIDER=anthropic python scripts/run_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragcite.eval import gate, load_golden_set, run_eval, write_reports
from ragcite.generation.llm import build_llm_client
from ragcite.index import IndexStore
from ragcite.pipeline import RagPipeline


def main() -> None:
    store = IndexStore.load()
    pipeline = RagPipeline(store, llm=build_llm_client())
    golden_set = load_golden_set()

    print(f"Running eval over {len(golden_set)} questions with provider={pipeline.llm.provider} "
          f"model={pipeline.llm.model} ...")
    scores, summary = run_eval(pipeline, golden_set)
    write_reports(scores, summary, ROOT / "eval")

    print(f"Faithfulness:      {summary['faithfulness']:.3f}")
    print(f"Citation accuracy: {summary['citation_accuracy']:.3f}")
    print(f"Context recall:    {summary['context_recall']:.3f}")
    print(f"Context precision: {summary['context_precision']:.3f}")
    print(f"Grounded rate:     {summary['grounded_rate']:.3f}")
    print(f"Total cost:        ${summary['total_cost_usd']:.4f}")
    print(f"Avg latency:       {summary['avg_latency_s']:.3f}s")

    passed, failures = gate(summary)
    if not passed:
        print("\nEVAL GATE FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nEVAL GATE PASSED")


if __name__ == "__main__":
    main()
