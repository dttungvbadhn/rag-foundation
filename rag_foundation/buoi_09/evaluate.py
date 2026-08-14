"""Retrieval-only evaluator for Buổi 09.

Real evaluation may call query expansion, embeddings and the reranker only when
the user explicitly runs the CLI. Answer generation is never called here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

try:
    from . import advanced_rag as advanced
    from . import hierarchical_rag as pipeline
except ImportError:
    import advanced_rag as advanced
    import hierarchical_rag as pipeline


EVAL_PATH = Path(__file__).resolve().parent / "eval" / "questions.json"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
QUESTION_TYPES = {"exact", "paraphrase", "multi_aspect", "hierarchy_context", "out_of_scope"}


class EvaluationError(ValueError):
    """Safe evaluation data/config error."""


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float | None:
    if not relevant_ids:
        return None
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item_id
              in enumerate(ranked_ids[:k], start=1) if item_id in relevant_ids)
    ideal_count = min(k, len(relevant_ids))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def load_questions(
    path: Path, registry_children: list[dict[str, Any]],
    registry_parents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strictly validate provisional labels against the current hierarchy store."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Không đọc được evaluation data: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise EvaluationError("Evaluation data phải là list không rỗng")
    child_ids = {item["child_id"] for item in registry_children}
    parent_ids = {item["parent_id"] for item in registry_parents}
    seen = set()
    required = {"question_id", "question", "question_type", "relevant_child_ids",
                "relevant_parent_ids", "needs_human_review", "notes"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or set(item) != required:
            raise EvaluationError(f"Question record {index} sai schema")
        if item["question_id"] in seen:
            raise EvaluationError(f"Trùng question_id: {item['question_id']}")
        seen.add(item["question_id"])
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise EvaluationError(f"{item['question_id']}: question rỗng")
        if item["question_type"] not in QUESTION_TYPES:
            raise EvaluationError(f"{item['question_id']}: question_type không hợp lệ")
        if item["needs_human_review"] is not True:
            raise EvaluationError(f"{item['question_id']}: nhãn ban đầu phải needs_human_review=true")
        stale_children = set(item["relevant_child_ids"]) - child_ids
        stale_parents = set(item["relevant_parent_ids"]) - parent_ids
        if stale_children or stale_parents:
            raise EvaluationError(
                f"{item['question_id']}: stale labels child={sorted(stale_children)}, "
                f"parent={sorted(stale_parents)}"
            )
    return data


def _ranked_units(result: dict[str, Any], child_to_parent: dict[str, str]) -> tuple[list[str], list[str], set[str]]:
    mode = result["mode"]
    candidates = result.get("candidates", [])
    if mode.endswith("parent"):
        parent_ids = [item["parent_id"] for item in candidates]
        child_ids = [child_id for item in candidates for child_id in item.get("supporting_child_ids", [])]
        sources = {item["source"] for item in candidates if item.get("source")}
    else:
        child_ids = [item.get("child_id") or item.get("chunk_id") for item in candidates]
        parent_ids = list(dict.fromkeys(child_to_parent[item] for item in child_ids if item in child_to_parent))
        sources = {item["source"] for item in candidates if item.get("source")}
    return child_ids, parent_ids, sources


def evaluate_questions(
    questions: list[dict[str, Any]], k: int,
    hierarchy_config: Any, advanced_config: Any, chunks: list[dict[str, Any]],
    registry_children: list[dict[str, Any]], registry_parents: list[dict[str, Any]],
    retrieval_fn: Any = pipeline.retrieve_complete_mode,
    retrieval_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare four modes retrieval-only; never call answer generation."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise EvaluationError("K phải là integer dương")
    child_to_parent = {item["child_id"]: item["parent_id"] for item in registry_children}
    parent_by_id = {item["parent_id"]: item for item in registry_parents}
    per_question, failures = [], []
    aggregate_values: dict[str, dict[str, list[float]]] = {
        mode: {} for mode in pipeline.ANSWER_MODES
    }
    identities = None
    for question in questions:
        mode_results = {}
        for mode in pipeline.ANSWER_MODES:
            try:
                result = retrieval_fn(
                    question["question"], mode, hierarchy_config, advanced_config,
                    chunks, **(retrieval_options or {}),
                )
                child_ids, parent_ids, sources = _ranked_units(result, child_to_parent)
                relevant_children = set(question["relevant_child_ids"])
                relevant_parents = set(question["relevant_parent_ids"])
                child_recall = recall_at_k(child_ids, relevant_children, k)
                parent_recall = recall_at_k(parent_ids, relevant_parents, k)
                ranking_ids = parent_ids if mode.endswith("parent") else child_ids
                relevant_ranking = relevant_parents if mode.endswith("parent") else relevant_children
                mrr = reciprocal_rank_at_k(ranking_ids, relevant_ranking, k)
                ndcg = ndcg_at_k(ranking_ids, relevant_ranking, k)
                context_chars = sum(len(item.get("text", "")) for item in result.get("candidates", []))
                child_chars = sum(len(item.get("text", "")) for item in result.get("child_hits", []))
                trace = result.get("trace", {})
                calls = trace.get("api_calls", {})
                metrics = {
                    "child_recall_at_k": child_recall,
                    "parent_recall_at_k": parent_recall,
                    "mrr_at_k": mrr, "ndcg_at_k": ndcg,
                    "unique_relevant_parents_retrieved": len(set(parent_ids) & relevant_parents),
                    "unique_sources_retrieved": len(sources),
                    "query_count": len(result.get("query_set", {}).get("queries", [])),
                    "child_union_count": len(result.get("child_hits", [])),
                    "context_chars": context_chars,
                    "expansion_factor": context_chars / child_chars if child_chars else 0.0,
                    "latency_ms": trace.get("total_latency_ms", 0.0),
                    "query_generation_calls": calls.get("generation_expansion", 0),
                    "embedding_calls": calls.get("embedding", 0),
                }
                mode_results[mode] = {"status": result.get("status"), "metrics": metrics,
                                      "child_ids": child_ids[:k], "parent_ids": parent_ids[:k],
                                      "warnings": result.get("warnings", [])}
                identities = identities or trace.get("identity")
                for name, value in metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        aggregate_values[mode].setdefault(name, []).append(float(value))
                if result.get("status") not in {"ready", "partial", "multi_query_partial"}:
                    failures.append({"question_id": question["question_id"],
                                     "mode": mode, "status": result.get("status")})
            except Exception as exc:
                mode_results[mode] = {"status": "failed", "error": type(exc).__name__}
                failures.append({"question_id": question["question_id"],
                                 "mode": mode, "status": "failed", "error": type(exc).__name__})
        per_question.append({"question_id": question["question_id"],
                             "question_type": question["question_type"],
                             "needs_human_review": question["needs_human_review"],
                             "modes": mode_results})
    aggregates = {}
    for mode, metric_values in aggregate_values.items():
        aggregates[mode] = {}
        for name, values in metric_values.items():
            aggregates[mode][f"mean_{name}"] = statistics.fmean(values)
            if name == "latency_ms":
                aggregates[mode]["p50_latency_ms"] = statistics.median(values)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "k": k,
        "identity": identities, "question_count": len(questions),
        "per_question": per_question, "aggregate_metrics": aggregates,
        "failures": failures,
        "needs_human_review": any(item["needs_human_review"] for item in questions),
        "human_review_warning": (
            "Gold labels đang needs_human_review=true; không được dùng để tuyên bố mode thắng."
        ),
        "answer_generation_calls": 0,
    }


def write_report_atomic(report: dict[str, Any], reports_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    """Publish timestamped and latest report only after complete JSON validation."""
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["timestamp"].replace(":", "-").replace("+", "_")
    report_path = reports_dir / f"evaluation_{stamp}.json"
    latest_path = reports_dir / "latest_report.json"
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8"); temporary.replace(report_path)
    latest_temp = latest_path.with_name(f".{latest_path.name}.tmp")
    latest_temp.write_text(payload, encoding="utf-8"); latest_temp.replace(latest_path)
    return report_path, latest_path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Buổi 09 retrieval-only evaluator")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=EVAL_PATH)
    args = parser.parse_args()
    try:
        env = pipeline.ENV_PATH
        hierarchy_config = pipeline.load_hierarchy_config(env)
        advanced_config = advanced.load_advanced_config(env)
        store = pipeline.load_hierarchy_store(hierarchy_config)
        if store["status"] != "ready":
            raise EvaluationError(store["error"])
        questions = load_questions(args.questions, store["children"], store["parents"])
        chunks, _ = pipeline.load_hierarchical_chunks()
        report = evaluate_questions(
            questions, args.k, hierarchy_config, advanced_config, chunks,
            store["children"], store["parents"],
        )
        report_path, latest_path = write_report_atomic(report)
        print(json.dumps({"report": str(report_path), "latest": str(latest_path),
                          "aggregate_metrics": report["aggregate_metrics"],
                          "human_review_warning": report["human_review_warning"]},
                         ensure_ascii=False, indent=2))
        return 0
    except (EvaluationError, pipeline.HierarchyError, advanced.AdvancedConfigError) as exc:
        print(f"LỖI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
