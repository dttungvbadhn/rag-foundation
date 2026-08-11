"""Retrieval evaluator Buổi 08; không gọi generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

try:
    from . import advanced_rag as ar
except ImportError:  # Chạy trực tiếp: python evaluate.py ...
    import advanced_rag as ar


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = BASE_DIR / "eval" / "questions.json"
REPORTS_DIR = BASE_DIR / "reports"


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def mrr_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranking[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(ranking[:k], 1) if chunk_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation questions phải là list không rỗng")
    required = {"query_id", "question", "relevant_chunk_ids", "scope", "needs_human_review"}
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict) or not required <= set(item):
            raise ValueError(f"Evaluation question #{index} sai schema")
    return data


def evaluate_retrieval(
    questions: list[dict[str, Any]],
    modes: list[str],
    strategy: str,
    k: int,
    chunks: list[dict],
    config: ar.AdvancedConfig,
    retrieval: Any = ar.retrieve_mode,
) -> dict[str, Any]:
    """Đánh giá cùng corpus/query/k cho mọi mode; lỗi query được ghi rõ."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k phải là integer dương")
    details: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, list[float]]] = {
        mode: {"recall": [], "mrr": [], "ndcg": [], "latency": []} for mode in modes
    }
    for question in questions:
        relevant = set(question["relevant_chunk_ids"])
        for mode in modes:
            started = time.perf_counter()
            try:
                result = retrieval(question["question"], mode, strategy, chunks, config)
                if result["status"] == "reranker_unavailable":
                    raise RuntimeError("reranker_unavailable")
                ranking = [item["chunk_id"] for item in result["candidates"][:k]]
                latency = (time.perf_counter() - started) * 1000
                scores = {
                    "recall_at_k": recall_at_k(ranking, relevant, k),
                    "mrr_at_k": mrr_at_k(ranking, relevant, k),
                    "ndcg_at_k": ndcg_at_k(ranking, relevant, k),
                }
                aggregates[mode]["recall"].append(scores["recall_at_k"])
                aggregates[mode]["mrr"].append(scores["mrr_at_k"])
                aggregates[mode]["ndcg"].append(scores["ndcg_at_k"])
                aggregates[mode]["latency"].append(latency)
                details.append({"query_id": question["query_id"], "mode": mode,
                                "status": "ok", "ranking": ranking, **scores,
                                "latency_ms": latency})
            except Exception as exc:
                details.append({"query_id": question["query_id"], "mode": mode,
                                "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    metrics = {}
    for mode, values in aggregates.items():
        latencies = values["latency"]
        metrics[mode] = {
            f"recall@{k}": statistics.mean(values["recall"]) if values["recall"] else None,
            f"mrr@{k}": statistics.mean(values["mrr"]) if values["mrr"] else None,
            f"ndcg@{k}": statistics.mean(values["ndcg"]) if values["ndcg"] else None,
            "latency_mean_ms": statistics.mean(latencies) if latencies else None,
            "latency_p50_ms": statistics.median(latencies) if latencies else None,
            "successful_queries": len(latencies),
        }
    needs_review = any(item["needs_human_review"] for item in questions)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "strategy": strategy,
        "k": k, "modes": modes, "embedding_model": config.embedding_model,
        "embedding_dim": config.embedding_dim, "reranker_model": config.reranker_model,
        "needs_human_review": needs_review,
        "warnings": (["Gold labels còn needs_human_review=true; không tuyên bố mode chiến thắng chính thức."] if needs_review else []),
        "metrics": metrics, "details": details,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Advanced RAG retrieval evaluator")
    parser.add_argument("--strategy", choices=("hierarchical", "semantic", "fixed-size"), default="hierarchical")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    args = parser.parse_args()
    try:
        config = ar.load_advanced_config(ar.ENV_PATH)
        chunks, _ = ar.load_chunks(strategy=args.strategy)
        report = evaluate_retrieval(load_questions(args.questions), list(ar.ANSWER_MODES), args.strategy, args.k, chunks, config)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = REPORTS_DIR / f"evaluation_{stamp}.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu report: {output}")
        print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"LỖI: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
