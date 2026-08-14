"""Run the same QA set at 0/1/2 hops and write a reproducible Markdown report."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluate import load_questions
from graph_retrieval import RetrievalConfig, VietnameseMSMARCOEmbedder, search_context
from qa_pipeline import QAConfig, answer_question, format_context


PROJECT_DIR = Path(__file__).resolve().parent


def run_comparison(
    questions: list[dict[str, Any]], *, hops: tuple[int, ...] = (0, 1, 2), top_k: int = 5,
    generate: bool = True, retriever: Callable[..., dict[str, Any]] = search_context,
    answerer: Callable[..., dict[str, Any]] = answer_question,
) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for hop in hops:
            started = time.perf_counter()
            try:
                retrieval = retriever(
                    question["question"],
                    retrieval_config=RetrievalConfig(top_k=top_k, max_hops=hop),
                )
                context, _, truncated = format_context(retrieval)
                qa = {"status": "not_run", "answer": None, "generation_call_count": 0}
                if generate:
                    qa = answerer(question["question"], retrieval=retrieval, qa_config=QAConfig.from_env())
                related = retrieval.get("related", [])
                found_relationships = sorted({
                    relationship for item in related
                    for relationship in item.get("relationship_path", [])
                })
                rows.append({
                    "question_id": question["question_id"], "hops": hop, "status": "ok",
                    "seed_count": len(retrieval.get("seeds", [])), "related_count": len(related),
                    "context_chars": len(context), "relationships": found_relationships,
                    "qa_status": qa.get("status"), "answer": qa.get("answer"),
                    "generation_calls": qa.get("generation_call_count", 0),
                    "context_truncated": truncated,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                })
            except Exception as exc:
                rows.append({"question_id": question["question_id"], "hops": hop,
                             "status": "failed", "error_type": type(exc).__name__,
                             "error": str(exc), "generation_calls": 0,
                             "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    return rows


def render_markdown(questions: list[dict[str, Any]], rows: list[dict[str, Any]], top_k: int,
                    generate: bool = True) -> str:
    retrieval_complete = all(row["status"] == "ok" for row in rows)
    qa_complete = generate and all(row.get("qa_status") == "answered" for row in rows)
    if qa_complete:
        run_status = "COMPLETED"
    elif retrieval_complete and not generate:
        run_status = "RETRIEVAL COMPLETED / QA NOT RUN"
    else:
        run_status = "NOT RUN / INCOMPLETE"
    lines = [
        "# So sánh hệ thống QA theo số bước nhảy",
        "",
        f"- Thời điểm UTC: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Thiết lập cố định: `top_k={top_k}`, cùng model embedding, vector index và prompt.",
        "- Biến độc lập: `max_hops ∈ {0, 1, 2}`.",
        f"- Trạng thái: **{run_status}**.",
        "",
        "## Kết quả định lượng",
        "",
        "| Câu | Hop | Trạng thái | Seed | Related | Context chars | Quan hệ | Latency ms | Gemini calls |",
        "|---|---:|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question_id']} | {row['hops']} | {row.get('qa_status', row['status'])} | "
            f"{row.get('seed_count', '—')} | {row.get('related_count', '—')} | "
            f"{row.get('context_chars', '—')} | {', '.join(row.get('relationships', [])) or '—'} | "
            f"{row['latency_ms']} | {row.get('generation_calls', 0)} |"
        )
    lines.extend(["", "## Câu trả lời theo từng cấu hình", ""])
    by_id = {item["question_id"]: item for item in questions}
    for question_id, question in by_id.items():
        lines.extend([f"### {question_id}", "", question["question"], ""])
        for row in (item for item in rows if item["question_id"] == question_id):
            answer = row.get("answer") or f"NOT RUN — {row.get('error_type', row.get('qa_status', 'unknown'))}: {row.get('error', '')}".strip()
            lines.extend([f"**{row['hops']} hop:** {answer}", ""])
    lines.extend([
        "## Kết luận",
        "",
        ("So sánh đã chạy hoàn chỉnh. Chỉ kết luận lợi ích multi-hop khi câu trả lời ở hop lớn hơn "
         "có thêm bằng chứng liên quan đúng và được người đánh giá xác nhận."
         if qa_complete else
         "Retrieval 0/1/2 hop đã chạy hoàn chỉnh; phần QA chưa chạy. Có thể đánh giá mức mở rộng "
         "context và quan hệ, nhưng chưa thể so sánh chất lượng câu trả lời Gemini."
         if retrieval_complete and not generate else
         "Chưa thể chứng minh hiệu quả multi-hop bằng kết quả thực nghiệm vì ít nhất một cấu hình "
         "không chạy hoàn chỉnh. Không sử dụng dữ liệu giả để thay thế kết quả runtime."),
        "",
        "Các câu trả lời và quan hệ thu được vẫn cần chuyên gia kiểm tra; số context lớn hơn không tự động đồng nghĩa chính xác hơn.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "qa_comparison.md")
    args = parser.parse_args()
    questions = load_questions()
    shared_embedder = VietnameseMSMARCOEmbedder()
    def shared_retriever(question: str, *, retrieval_config: RetrievalConfig):
        return search_context(question, retrieval_config=retrieval_config, embedder=shared_embedder)
    rows = run_comparison(
        questions, top_k=args.top_k, generate=not args.retrieval_only,
        retriever=shared_retriever,
    )
    args.output.write_text(
        render_markdown(questions, rows, args.top_k, generate=not args.retrieval_only), encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    return 0 if all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
