from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

from ragcite.config import ROOT_DIR
from ragcite.eval.metrics import QuestionScore, score_question
from ragcite.pipeline import RagPipeline

DEFAULT_GATE_THRESHOLDS = {
    "faithfulness": 0.55,
    "citation_accuracy": 0.85,
    "context_recall": 0.70,
    "grounded_rate": 0.90,
}


def load_golden_set(path: Path | None = None) -> list[dict]:
    path = path or (ROOT_DIR / "eval" / "golden_set.json")
    return json.loads(Path(path).read_text())


def run_eval(
    pipeline: RagPipeline,
    golden_set: list[dict] | None = None,
) -> tuple[list[QuestionScore], dict]:
    golden_set = golden_set if golden_set is not None else load_golden_set()

    scores: list[QuestionScore] = []
    t0 = time.perf_counter()
    for item in golden_set:
        result = pipeline.answer(item["question"])
        scores.append(score_question(item, result))
    wall_s = time.perf_counter() - t0

    def mean(attr: str) -> float:
        values = [getattr(s, attr) for s in scores]
        return statistics.fmean(values) if values else 0.0

    grounded_rate = statistics.fmean([1.0 if s.grounded else 0.0 for s in scores]) if scores else 0.0
    summary = {
        "n_questions": len(scores),
        "faithfulness": mean("faithfulness"),
        "answer_relevancy": mean("answer_relevancy"),
        "context_precision": mean("context_precision"),
        "context_recall": mean("context_recall"),
        "citation_accuracy": mean("citation_accuracy"),
        "keyword_recall": mean("keyword_recall"),
        "grounded_rate": grounded_rate,
        "total_cost_usd": sum(s.cost_usd for s in scores),
        "avg_cost_usd": mean("cost_usd"),
        "avg_latency_s": mean("latency_s"),
        "p95_latency_s": (
            statistics.quantiles([s.latency_s for s in scores], n=20)[18] if len(scores) >= 20 else max(
                (s.latency_s for s in scores), default=0.0
            )
        ),
        "wall_clock_s": wall_s,
        "provider": pipeline.llm.provider,
        "model": pipeline.llm.model,
    }
    return scores, summary


def gate(summary: dict, thresholds: dict | None = None) -> tuple[bool, list[str]]:
    thresholds = thresholds or DEFAULT_GATE_THRESHOLDS
    failures = []
    for metric, minimum in thresholds.items():
        value = summary.get(metric, 0.0)
        if value < minimum:
            failures.append(f"{metric}={value:.3f} < required {minimum:.3f}")
    return (len(failures) == 0), failures


def to_markdown_table(scores: list[QuestionScore], summary: dict) -> str:
    lines = [
        "# RAGAS-style Eval Report",
        "",
        (
            f"Provider/model: `{summary['provider']}/{summary['model']}`  |  "
            f"Questions: {summary['n_questions']}  |  "
            f"Total cost: ${summary['total_cost_usd']:.4f}  |  "
            f"Avg latency: {summary['avg_latency_s']:.3f}s  |  "
            f"p95 latency: {summary['p95_latency_s']:.3f}s"
        ),
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Faithfulness | {summary['faithfulness']:.3f} |",
        f"| Answer Relevancy | {summary['answer_relevancy']:.3f} |",
        f"| Context Precision | {summary['context_precision']:.3f} |",
        f"| Context Recall | {summary['context_recall']:.3f} |",
        f"| Citation Accuracy | {summary['citation_accuracy']:.3f} |",
        f"| Keyword Recall | {summary['keyword_recall']:.3f} |",
        f"| Grounded Rate | {summary['grounded_rate']:.3f} |",
        f"| Avg Cost / Query | ${summary['avg_cost_usd']:.6f} |",
        f"| Avg Latency / Query | {summary['avg_latency_s']:.3f}s |",
        "",
        "## Per-question detail",
        "",
        "| id | faithfulness | ctx precision | ctx recall | citation acc | grounded | cost | latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        lines.append(
            f"| {s.id} | {s.faithfulness:.2f} | {s.context_precision:.2f} | {s.context_recall:.2f} | "
            f"{s.citation_accuracy:.2f} | {'yes' if s.grounded else 'NO'} | ${s.cost_usd:.6f} | {s.latency_s:.3f}s |"
        )
    return "\n".join(lines) + "\n"


def write_reports(scores: list[QuestionScore], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "scores": [asdict(s) for s in scores]}, indent=2)
    )
    (out_dir / "report.md").write_text(to_markdown_table(scores, summary))
