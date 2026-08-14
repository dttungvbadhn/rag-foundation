"""Evaluate Graph RAG retrieval and optional grounded QA on five legal questions."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from graph_retrieval import RetrievalConfig, search_context
from qa_pipeline import QAConfig, answer_question


PROJECT_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = PROJECT_DIR / "eval" / "questions.json"
REPORTS_DIR = PROJECT_DIR / "reports"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if len(data) != 5 or any(not item.get("question_id") or not item.get("question") for item in data):
        raise ValueError("Evaluation set phai co dung 5 cau hoi hop le")
    return data


def _relationship_coverage(expected: list[str], related: list[dict[str, Any]]) -> float:
    if not expected:
        return 1.0
    found = {rel for row in related for rel in row.get("relationship_path", [])}
    return len(set(expected) & found) / len(set(expected))


def evaluate(
    questions: list[dict[str, Any]], *, top_k: int = 5, max_hops: int = 2,
    generate: bool = False, retriever: Callable[..., dict[str, Any]] = search_context,
    answerer: Callable[..., dict[str, Any]] = answer_question,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in questions:
        started = time.perf_counter()
        try:
            retrieval = retriever(item["question"], retrieval_config=RetrievalConfig(top_k=top_k, max_hops=max_hops))
            retrieval_ms = (time.perf_counter() - started) * 1000
            seeds, related = retrieval.get("seeds", []), retrieval.get("related", [])
            qa = {"status": "not_run", "generation_call_count": 0, "answer": None}
            if generate:
                qa = answerer(item["question"], retrieval=retrieval, qa_config=QAConfig.from_env())
            rows.append({
                "question_id": item["question_id"], "question": item["question"],
                "scenario": item.get("scenario"), "status": "ok",
                "seed_count": len(seeds), "related_count": len(related),
                "context_count": len(seeds) + len(related),
                "max_observed_hop": max((row.get("hop", 0) for row in related), default=0),
                "expected_relationships": item.get("expected_relationships", []),
                "relationship_coverage": _relationship_coverage(item.get("expected_relationships", []), related),
                "seed_ids": [row.get("id") for row in seeds],
                "related_ids": [row.get("id") for row in related],
                "retrieval_latency_ms": round(retrieval_ms, 3),
                "qa_status": qa.get("status"), "generation_call_count": qa.get("generation_call_count", 0),
                "answer": qa.get("answer"), "needs_human_review": item.get("needs_human_review", True),
            })
        except Exception as exc:
            rows.append({"question_id": item["question_id"], "question": item["question"],
                         "scenario": item.get("scenario"), "status": "failed",
                         "error_type": type(exc).__name__, "error": str(exc),
                         "generation_call_count": 0, "needs_human_review": True})
    successful = [row for row in rows if row["status"] == "ok"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"top_k": top_k, "max_hops": max_hops, "generate": generate},
        "summary": {
            "question_count": len(rows), "successful": len(successful),
            "failed": len(rows) - len(successful),
            "mean_seed_count": statistics.mean([r["seed_count"] for r in successful]) if successful else 0,
            "mean_related_count": statistics.mean([r["related_count"] for r in successful]) if successful else 0,
            "mean_relationship_coverage": statistics.mean([r["relationship_coverage"] for r in successful]) if successful else 0,
            "mean_retrieval_latency_ms": statistics.mean([r["retrieval_latency_ms"] for r in successful]) if successful else 0,
            "human_review_required": True,
        },
        "questions": rows,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hops", type=int, default=2)
    parser.add_argument("--generate", action="store_true", help="Cho phep goi Gemini")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "latest_report.json")
    args = parser.parse_args()
    report = evaluate(load_questions(), top_k=args.top_k, max_hops=args.max_hops, generate=args.generate)
    write_report(report, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
