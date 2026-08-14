"""Streamlit UI for Buổi 09 Multi-query and Parent–Child Retrieval.

Importing this module is side-effect free. Network/model/index actions only run
after an explicit Streamlit button click.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from . import advanced_rag as advanced
    from . import hierarchical_rag as pipeline
except ImportError:  # streamlit run app.py
    import advanced_rag as advanced
    import hierarchical_rag as pipeline


PAGE_TITLE = "RAG Foundation — Buổi 09: Multi-query & Parent–Child Retrieval"
PIPELINE_SUBTITLE = (
    "Query fan-out → Hybrid per query → Cross-query RRF → "
    "Parent expansion → Parent rerank"
)


STATUS_GUIDANCE = {
    "hierarchy_not_ready": "Hãy kiểm tra hierarchy status và chủ động build lại hierarchy.",
    "collection_not_ready": "Hãy chạy Prepare semantic sau khi kiểm tra cấu hình và API key.",
    "query_generation_unavailable": "Kiểm tra Gemini key/model; single mode vẫn có thể dùng riêng.",
    "multi_query_partial": "Một số query mở rộng lỗi; kết quả hiện chỉ là partial.",
    "partial": "Một phần retrieval lỗi; hãy xem Query Fan-out và warnings.",
    "reranker_unavailable": "Kiểm tra Internet, dung lượng cache, RAM và cấu hình reranker.",
    "insufficient_evidence": "Không có evidence đạt gate; không gọi answer generation.",
    "generation_error": "Retrieval đã chạy nhưng answer generation lỗi.",
    "retrieval_only": "Đã có evidence nhưng chưa tạo được câu trả lời tổng hợp.",
}


def status_message(status: str) -> dict[str, str]:
    """Map pipeline status to safe UI severity and actionable guidance."""
    severity = {
        "answered": "success", "ready": "success", "partial": "warning",
        "multi_query_partial": "warning", "insufficient_evidence": "warning",
        "retrieval_only": "warning",
    }.get(status, "error")
    return {"severity": severity,
            "message": STATUS_GUIDANCE.get(status, f"Trạng thái pipeline: {status}")}


def format_citation(citation: dict[str, Any]) -> str:
    """Format mapped citation metadata without trusting model-generated source data."""
    start, end = citation.get("page_start"), citation.get("page_end")
    pages = str(start) if start == end else f"{start}-{end}"
    parent = f", parent: {citation['parent_id']}" if citation.get("parent_id") else ""
    anchor = f", anchor: {citation['anchor_child_id']}" if citation.get("anchor_child_id") else ""
    return f"[{citation.get('evidence_id', '?')}] {citation.get('source', '?')} — tr. {pages}{parent}{anchor}"


def build_query_child_matrix(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rows child × query with rank cells and fusion support fields."""
    queries = result.get("query_set", {}).get("queries", [])
    query_ids = [item["query_id"] for item in queries]
    rows = []
    for child in result.get("child_hits", result.get("children", [])):
        ranks = child.get("per_query_ranks", {})
        row = {"child_id": child.get("child_id") or child.get("chunk_id")}
        row.update({query_id: ranks.get(query_id, "—") for query_id in query_ids})
        row["support_query_count"] = child.get("support_query_count", 1)
        row["MQ-RRF score"] = child.get("multi_query_rrf_score")
        rows.append(row)
    return rows


def build_parent_tree_data(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize parent/child/rank data for an explainable UI tree."""
    mapping_source = result.get("child_hits", [])
    child_by_id = {item.get("child_id") or item.get("chunk_id"): item
                   for item in mapping_source}
    parents = result.get("accepted_evidence") or result.get("parent_candidates", [])
    rows = []
    for parent in parents:
        children = []
        for child_id in parent.get("supporting_child_ids", []):
            child = child_by_id.get(child_id, {})
            children.append({
                "child_id": child_id,
                "query_ids": child.get("support_query_ids", []),
                "query_ranks": child.get("per_query_ranks", {}),
                "snippet": child.get("text", "")[:240],
                "anchor": child_id == parent.get("anchor_child_id"),
            })
        rows.append({
            "parent_id": parent.get("parent_id"), "source": parent.get("source"),
            "page_start": parent.get("page_start"), "page_end": parent.get("page_end"),
            "structural_path": parent.get("structural_path", {}),
            "parent_rank": parent.get("parent_rank"),
            "parent_rerank_rank": parent.get("parent_rerank_rank"),
            "parent_rrf_score": parent.get("parent_rrf_score"),
            "parent_rerank_score": parent.get("parent_rerank_score"),
            "text": parent.get("text", ""), "children": children,
            "ambiguous": bool(parent.get("ambiguous")),
            "warnings": list(parent.get("warnings", [])),
        })
    return rows


def build_comparison_row(mode: str, result: dict[str, Any]) -> dict[str, Any]:
    """Summarize one retrieval-only mode without declaring a winner."""
    candidates = result.get("candidates", [])
    is_parent = mode.endswith("parent")
    child_hits = result.get("child_hits", [])
    sources = {item.get("source") for item in candidates if item.get("source")}
    articles = {
        item.get("structural_path", {}).get("article") for item in candidates
        if item.get("structural_path", {}).get("article")
    }
    context_chars = sum(len(item.get("text", "")) for item in candidates)
    child_chars = sum(len(item.get("text", "")) for item in child_hits)
    trace = result.get("trace", {})
    calls = trace.get("api_calls", {})
    evidence_ids = [item.get("parent_id") if is_parent else item.get("chunk_id")
                    for item in candidates]
    return {
        "mode": mode, "status": result.get("status"),
        "unit_type": "parent" if is_parent else "child",
        "final_evidence_ids": ", ".join(filter(None, evidence_ids)),
        "rank_fields": "parent_rank→parent_rerank_rank" if is_parent else "fused_rank→rerank_rank",
        "unique_sources": len(sources), "unique_articles": len(articles),
        "retrieved_child_count": len(child_hits),
        "expanded_parent_count": len(candidates) if is_parent else 0,
        "context_chars": context_chars,
        "expansion_factor": context_chars / child_chars if child_chars else 0.0,
        "latency_ms": trace.get("total_latency_ms", 0.0),
        "Generation calls": calls.get("generation_expansion", 0)
                            + calls.get("generation_answer", 0),
        "Embedding calls": calls.get("embedding", 0),
        "warnings": "; ".join(result.get("warnings", [])),
    }


def _config_paths() -> Path:
    return pipeline.ENV_PATH if pipeline.ENV_PATH.is_file() else pipeline.ENV_EXAMPLE_PATH


def _latest_report() -> dict[str, Any] | None:
    reports = sorted((pipeline.BUOI_09_DIR / "reports").glob("*.json"),
                     key=lambda path: path.stat().st_mtime, reverse=True)
    if not reports:
        return None
    try:
        return json.loads(reports[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid_report", "file": reports[0].name}


def _render_status(status: str) -> None:
    mapped = status_message(status)
    getattr(st, mapped["severity"])(f"{status}: {mapped['message']}")


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: Không thể hoàn tất action. Kiểm tra cấu hình và trạng thái hệ thống."


def _sidebar_configs() -> tuple[str, pipeline.HierarchyConfig, Any, dict[str, Any]]:
    env_path = _config_paths()
    hierarchy_config = pipeline.load_hierarchy_config(env_path)
    advanced_config = advanced.load_advanced_config(env_path)
    st.sidebar.header("Cấu hình pipeline")
    mode = st.sidebar.selectbox("Mode", pipeline.ANSWER_MODES, index=3)
    multi_count = st.sidebar.slider("MULTI_QUERY_COUNT", 1, 5,
                                    hierarchy_config.multi_query_count)
    per_query = st.sidebar.slider("PER_QUERY_CANDIDATES", 1, 100,
                                  hierarchy_config.per_query_candidates)
    parent_candidates = st.sidebar.slider("PARENT_CANDIDATES", 1, 100,
                                          hierarchy_config.parent_candidates)
    final_parent = st.sidebar.slider("FINAL_PARENT_TOP_K", 1, parent_candidates,
                                     min(hierarchy_config.final_parent_top_k,
                                         parent_candidates))
    min_score = st.sidebar.slider("RERANK_MIN_SCORE", 0.0, 1.0,
                                  float(advanced_config.rerank_min_score), 0.01)
    hierarchy_config = replace(
        hierarchy_config, multi_query_count=multi_count,
        per_query_candidates=per_query, parent_candidates=parent_candidates,
        final_parent_top_k=final_parent,
    )
    advanced_config = replace(advanced_config, rerank_min_score=min_score)

    st.sidebar.caption("Strategy: hierarchical (cố định)")
    st.sidebar.write(f"Gemini key: {'Có' if advanced_config.api_key else 'Thiếu'}")
    st.sidebar.write(f"Embedding: {advanced_config.embedding_model} / {advanced_config.embedding_dim}")
    st.sidebar.write(f"Generation: {advanced_config.generation_model}")
    st.sidebar.write(f"Reranker: {advanced_config.reranker_model}")

    hierarchy = pipeline.load_hierarchy_store(hierarchy_config)
    if hierarchy["status"] == "ready":
        ambiguous = sum(bool(item.get("ambiguous")) for item in hierarchy["children"])
        hierarchy_summary = {"status": "ready", "children": len(hierarchy["children"]),
                             "parents": len(hierarchy["parents"]), "ambiguous": ambiguous}
    else:
        hierarchy_summary = {"status": "stale/missing", "children": 0,
                             "parents": 0, "ambiguous": 0}
    st.sidebar.write(f"Hierarchy: {hierarchy_summary['status']}")
    st.sidebar.write(f"Children/parents: {hierarchy_summary['children']}/{hierarchy_summary['parents']}")
    st.sidebar.write(f"Ambiguous: {hierarchy_summary['ambiguous']}")

    try:
        semantic = advanced.advanced_status("hierarchical", advanced_config)
        st.sidebar.write(
            f"Collection: {'ready' if semantic['semantic_collection_exists'] else 'missing'} "
            f"({semantic['semantic_collection_count']} records)"
        )
    except Exception:
        semantic = {"semantic_collection_exists": False, "semantic_collection_count": 0}
        st.sidebar.write("Collection: unavailable")
    return mode, hierarchy_config, advanced_config, semantic


def _render_actions(hierarchy_config: pipeline.HierarchyConfig, advanced_config: Any) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("Actions chủ động")
    confirmed = st.sidebar.checkbox("Tôi xác nhận action có thể ghi storage/gọi API")
    if st.sidebar.button("Build hierarchy", disabled=not confirmed):
        try:
            registry = pipeline.build_registry(hierarchy_config)
            pipeline.write_registry(registry)
            st.sidebar.success(f"Đã build {len(registry['children'])} child / {len(registry['parents'])} parent")
        except Exception as exc:
            st.sidebar.error(_safe_error(exc))
    if st.sidebar.button("Prepare semantic", disabled=not confirmed):
        try:
            result = advanced.prepare_semantic("hierarchical", advanced_config)
            st.sidebar.success(f"Semantic index: {result.get('collection', result.get('collection_name', 'done'))}")
        except Exception as exc:
            st.sidebar.error(_safe_error(exc))


def _render_answer(result: dict[str, Any]) -> None:
    _render_status(result.get("status", "unknown"))
    st.markdown(result.get("answer", "Chưa có câu trả lời."))
    if result.get("citations"):
        st.subheader("Citations")
        for citation in result["citations"]:
            st.write(format_citation(citation))
            if citation.get("ambiguous") or citation.get("warnings"):
                st.warning("; ".join(citation.get("warnings", [])) or "Hierarchy ambiguous")
    for warning in result.get("warnings", []):
        st.warning(warning)
    calls = result.get("trace", {}).get("api_calls", {})
    columns = st.columns(3)
    columns[0].metric("Tổng latency (ms)", f"{result.get('trace', {}).get('total_latency_ms', 0):.1f}")
    columns[1].metric("Generation calls", calls.get("generation_expansion", 0)
                      + calls.get("generation_answer", 0))
    columns[2].metric("Embedding calls", calls.get("embedding", 0))


def _render_fanout(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("Chưa có lần chạy. Query fan-out chỉ xuất hiện sau khi bấm chạy.")
        return
    query_set = result.get("query_set") or {}
    trace = result.get("trace", {}).get("child_retrieval", result.get("trace", {}))
    latencies = trace.get("retrieval_latency_ms", {})
    counts = trace.get("result_count_by_query", {})
    for item in query_set.get("queries", []):
        label = "Q0 · ORIGINAL" if item["query_id"] == "Q0" else f"{item['query_id']} · GENERATED"
        with st.container(border=True):
            st.markdown(f"**{label}** — focus: `{item.get('focus', 'unknown')}`")
            st.write(item["text"])
            st.caption(
                f"validation=valid · results={counts.get(item['query_id'], '—')} · "
                f"latency={latencies.get(item['query_id'], 0):.2f} ms"
            )
    st.subheader("Ma trận Query × Child")
    matrix = build_query_child_matrix(result)
    st.dataframe(matrix, use_container_width=True, hide_index=True) if matrix else st.info("Chưa có child hit.")


def _render_parent_tree(result: dict[str, Any] | None) -> None:
    if not result:
        st.info("Chưa có parent evidence.")
        return
    tree = build_parent_tree_data(result)
    if not tree:
        st.info("Mode hiện tại chưa trả parent.")
        return
    child_chars = sum(len(item.get("text", "")) for item in result.get("child_hits", []))
    parent_chars = sum(len(item["text"]) for item in tree)
    st.metric("Context expansion factor", f"{parent_chars / child_chars:.2f}×" if child_chars else "0×")
    for parent in tree:
        title = (f"{parent['parent_id']} · rank {parent['parent_rank']} → "
                 f"{parent['parent_rerank_rank']} · {parent['source']}")
        with st.expander(title, expanded=False):
            if parent["ambiguous"] or parent["warnings"]:
                st.warning("Ambiguous/warnings: " + "; ".join(parent["warnings"]))
            st.json(parent["structural_path"])
            st.write(f"Trang {parent['page_start']}-{parent['page_end']}")
            st.write(f"Parent-RRF: {parent['parent_rrf_score']} · Rerank: {parent['parent_rerank_score']}")
            st.markdown("**Supporting children**")
            for child in parent["children"]:
                marker = "⭐ anchor" if child["anchor"] else "child"
                st.write(f"- `{child['child_id']}` ({marker}) — queries={child['query_ids']}, ranks={child['query_ranks']}")
                if child["snippet"]:
                    st.caption(child["snippet"])
            st.markdown("**Parent text**")
            st.write(parent["text"])


def render_app() -> None:
    st.set_page_config(page_title="Buổi 09 · Multi-query Parent–Child", layout="wide")
    st.title(PAGE_TITLE)
    st.caption(PIPELINE_SUBTITLE)
    try:
        mode, hierarchy_config, advanced_config, _ = _sidebar_configs()
    except Exception as exc:
        st.error(_safe_error(exc)); return
    _render_actions(hierarchy_config, advanced_config)
    st.session_state.setdefault("last_answer", None)
    st.session_state.setdefault("last_compare", None)

    ask_tab, fanout_tab, parent_tab, compare_tab, evaluation_tab = st.tabs([
        "Ask Advanced RAG", "Query Fan-out", "Parent–Child Explorer",
        "Mode Comparison", "Evaluation",
    ])
    with ask_tab:
        question = st.text_area("Câu hỏi", placeholder="Nhập câu hỏi pháp lý cần tra cứu...")
        if st.button("Chạy Advanced RAG", type="primary"):
            if not question.strip():
                st.warning("Câu hỏi không được rỗng.")
            elif not advanced_config.api_key:
                st.error("Thiếu GEMINI_API_KEY trong .env; không gọi API.")
            else:
                try:
                    chunks, _ = pipeline.load_hierarchical_chunks()
                    with st.spinner("Đang fan-out, retrieval, rerank và grounding..."):
                        st.session_state.last_answer = pipeline.answer_complete(
                            question, mode, hierarchy_config, advanced_config, chunks
                        )
                except Exception as exc:
                    st.error(_safe_error(exc))
        if st.session_state.last_answer:
            _render_answer(st.session_state.last_answer)

    with fanout_tab:
        _render_fanout(st.session_state.last_answer)
    with parent_tab:
        _render_parent_tree(st.session_state.last_answer)
    with compare_tab:
        compare_question = st.text_area("Câu hỏi so sánh", key="compare_question")
        if st.button("So sánh bốn mode (không generation)"):
            if not compare_question.strip():
                st.warning("Câu hỏi không được rỗng.")
            elif not advanced_config.api_key:
                st.error("Thiếu GEMINI_API_KEY; semantic retrieval/query expansion không thể chạy.")
            else:
                try:
                    chunks, _ = pipeline.load_hierarchical_chunks()
                    with st.spinner("Đang chạy retrieval/rerank bốn mode..."):
                        st.session_state.last_compare = pipeline.compare_complete_modes(
                            compare_question, hierarchy_config, advanced_config, chunks
                        )
                except Exception as exc:
                    st.error(_safe_error(exc))
        if st.session_state.last_compare:
            rows = [build_comparison_row(mode_name, result) for mode_name, result
                    in st.session_state.last_compare["modes"].items()]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.info("Bảng chỉ mô tả retrieval/rerank; không tuyên bố mode thắng khi chưa có gold labels.")
    with evaluation_tab:
        report = _latest_report()
        if report is None:
            st.info("Chưa có report. UI không tự chạy evaluator khi render.")
        elif report.get("status") == "invalid_report":
            st.error("Latest report không đọc được.")
        else:
            if report.get("needs_human_review") or report.get("gold_labels_need_human_review"):
                st.warning("Gold labels có needs_human_review=true; chưa được chuyên gia duyệt.")
            metrics = report.get("metrics", report)
            wanted = ("child_recall_at_k", "parent_recall_at_k", "mrr_at_k",
                      "ndcg_at_k", "latency_ms", "context_chars")
            st.json({key: metrics.get(key) for key in wanted})


if __name__ == "__main__":
    render_app()
