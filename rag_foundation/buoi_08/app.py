"""Streamlit dashboard quan sát và so sánh pipeline Advanced RAG Buổi 08."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import advanced_rag as ar


APP_DIR = Path(__file__).resolve().parent
REPORTS_DIR = APP_DIR / "reports"
STRATEGIES = ("hierarchical", "semantic", "fixed-size")


@st.cache_resource(show_spinner=False)
def cached_corpus(strategy: str):
    """Cache corpus/BM25 theo strategy; không gọi API hoặc tạo index."""
    chunks, stats = ar.load_chunks(strategy=strategy)
    corpus = ar.build_bm25_corpus(chunks)
    return chunks, stats, corpus


def safe_error(exc: Exception) -> str:
    return f"Không thể thực hiện: {type(exc).__name__}. Hãy kiểm tra cấu hình và tài nguyên."


def evidence_card(item: dict) -> None:
    page = str(item["page_start"]) if item["page_start"] == item["page_end"] else f"{item['page_start']}-{item['page_end']}"
    mark = "Đạt gate" if item["accepted"] else "Không đạt gate"
    with st.expander(f"{item['evidence_id']} · {item['source']} · tr. {page} · {item['chunk_id']} · {mark}"):
        cols = st.columns(4)
        cols[0].write(f"BM25: rank={item['bm25_rank']}, score={item['bm25_score']}")
        cols[1].write(f"Semantic: rank={item['semantic_rank']}, distance={item['semantic_distance']}")
        cols[2].write(f"RRF: rank={item['fused_rank']}, score={item['rrf_score']}")
        cols[3].write(f"Rerank: rank={item['rerank_rank']}, score={item['rerank_score']}, Δ={item['rank_change']}")
        st.caption("Rerank score là điểm chuẩn hóa của model, không phải xác suất đúng.")
        st.write(item["text"])


def render_trace(trace: dict | None) -> None:
    if not trace:
        st.info("Chưa có pipeline trace.")
        return
    metrics = st.columns(5)
    for column, label, key in zip(
        metrics,
        ("BM25 candidates", "Semantic candidates", "Union / overlap", "Reranked", "Accepted"),
        ("bm25_candidates", "semantic_candidates", "union", "reranked", "accepted"),
    ):
        value = f"{trace.get('union', 0)} / {trace.get('overlap', 0)}" if key == "union" else trace.get(key, 0)
        column.metric(label, value)
    st.dataframe(
        [{"stage": stage, "latency_ms": round(float(value), 3)} for stage, value in trace.get("latency_ms", {}).items()],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("BM25 score cao hơn tốt hơn · cosine distance thấp hơn tốt hơn · RRF/rerank score cao hơn tốt hơn · rerank score không phải xác suất.")


st.set_page_config(page_title="Advanced RAG · Buổi 08", layout="wide")
st.title("Advanced RAG · Hybrid Search và Cross-Encoder Reranking")

config = ar._config_for_status()
strategy = st.sidebar.selectbox("Strategy", STRATEGIES)
mode = st.sidebar.selectbox("Retrieval mode", ar.ANSWER_MODES, index=3)
chunks, stats, _ = cached_corpus(strategy)
try:
    status = ar.advanced_status(strategy, config)
except Exception:
    status = {"semantic_collection_exists": False, "semantic_collection_count": 0,
              "semantic_collection_name": "không khả dụng", "reranker_cache_exists": False}

st.sidebar.subheader("Cấu hình pipeline")
st.sidebar.write(f"Final top-k: {config.final_top_k}")
st.sidebar.write(f"Candidate K: BM25={config.bm25_candidates}, semantic={config.semantic_candidates}")
st.sidebar.write(f"RRF: k={config.rrf_k}, weights={config.rrf_bm25_weight}/{config.rrf_semantic_weight}")
st.sidebar.write(f"Reranker: {config.reranker_model}")
st.sidebar.write(f"Device/cache: {config.rerank_device} / {'Có' if status['reranker_cache_exists'] else 'Thiếu'}")
st.sidebar.write(f"Rerank K/min score: {config.rerank_candidates} / {config.rerank_min_score}")
st.sidebar.write(f"Semantic: {status['semantic_collection_name']} ({status['semantic_collection_count']} chunks)")
st.sidebar.write(f"API key: {'Có' if config.api_key else 'Thiếu'}")

for key in ("answer_result", "compare_result"):
    st.session_state.setdefault(key, None)

answer_tab, compare_tab, trace_tab, evaluation_tab = st.tabs(
    ["Hỏi đáp Advanced RAG", "So sánh Retrieval", "Pipeline Trace", "Đánh giá"]
)

with answer_tab:
    question = st.text_area("Câu hỏi", key="answer_question")
    if st.button("Chạy Advanced RAG", type="primary"):
        if not question.strip():
            st.warning("Hãy nhập câu hỏi.")
        elif not config.api_key:
            st.warning("Thiếu GEMINI_API_KEY trong .env Buổi 08.")
        elif not status["semantic_collection_exists"] or not status["semantic_collection_count"]:
            st.warning("Chưa có semantic index. Hãy chủ động chạy command prepare-semantic.")
        else:
            try:
                with st.spinner("Đang chạy pipeline nhiều tầng..."):
                    st.session_state.answer_result = ar.answer_advanced(question, mode, strategy, chunks, config)
            except Exception as exc:
                st.error(safe_error(exc))
    result = st.session_state.answer_result
    if result:
        st.subheader(f"Trạng thái: {result['status']}")
        if result["status"] == "reranker_unavailable":
            st.warning("Reranker chưa khả dụng. Hãy kiểm tra Internet/disk/RAM, tải model bằng lần chạy chủ động rồi thử lại.")
        st.write(result["answer"])
        for warning in result["warnings"]:
            st.warning(warning)
        if result["citations"]:
            st.subheader("Citations")
            for citation in result["citations"]:
                st.write(citation["display"])
        st.subheader("Evidence")
        for item in result["evidence"]:
            evidence_card(item)

with compare_tab:
    compare_question = st.text_input("Câu hỏi so sánh", key="compare_question")
    if st.button("So sánh bốn mode"):
        if not compare_question.strip():
            st.warning("Hãy nhập câu hỏi.")
        elif not config.api_key or not status["semantic_collection_exists"]:
            st.warning("Compare cần API key và semantic index đã prepare; không có generation.")
        else:
            try:
                with st.spinner("Đang so sánh retrieval/rerank..."):
                    st.session_state.compare_result = ar.compare_modes(compare_question, strategy, chunks, config)
            except Exception as exc:
                st.error(safe_error(exc))
    comparison = st.session_state.compare_result
    if comparison:
        table = []
        for row in comparison["rows"]:
            ranks = row["ranks"]
            table.append({
                "chunk_id": row["chunk_id"], "bm25_rank": ranks.get("bm25"),
                "semantic_rank": ranks.get("semantic"), "fused_rank": ranks.get("hybrid"),
                "rerank_rank": ranks.get("hybrid_rerank"),
                "rank_change": row["rank_movement"].get("hybrid_rerank"),
                "final modes": ", ".join(row["modes"]),
            })
        st.dataframe(table, use_container_width=True, hide_index=True)
        panels = st.columns(4)
        for column, selected_mode in zip(panels, ar.ANSWER_MODES):
            column.subheader(selected_mode)
            for row in comparison["rows"]:
                if selected_mode in row["ranks"]:
                    column.write(f"#{row['ranks'][selected_mode]} · {row['chunk_id']}")

with trace_tab:
    render_trace(st.session_state.answer_result.get("trace") if st.session_state.answer_result else None)

with evaluation_tab:
    reports = sorted(REPORTS_DIR.glob("*.json"), reverse=True)
    if not reports:
        st.info("Chưa có evaluation report hợp lệ. Trang không tự chạy API/evaluator.")
    else:
        selected = st.selectbox("Report", reports, format_func=lambda path: path.name)
        try:
            report = json.loads(selected.read_text(encoding="utf-8"))
            if report.get("needs_human_review"):
                st.warning("Gold labels còn needs_human_review=true; không kết luận mode chiến thắng.")
            rows = []
            for report_mode, metrics in report.get("metrics", {}).items():
                rows.append({"mode": report_mode, **metrics})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            st.error(safe_error(exc))
