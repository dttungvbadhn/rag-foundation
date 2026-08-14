"""Streamlit UI for Vietnamese multi-hop Graph RAG QA."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from graph_retrieval import RetrievalConfig, VietnameseMSMARCOEmbedder, search_context
from neo4j_connection import Neo4jConfig, check_connection
from qa_pipeline import QAConfig, answer_question


st.set_page_config(page_title="Multi-hop Graph RAG", page_icon="🔗", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình MSMARCO tiếng Việt…")
def embedding_model() -> VietnameseMSMARCOEmbedder:
    return VietnameseMSMARCOEmbedder()


@st.cache_data(ttl=30, show_spinner=False)
def database_status() -> dict[str, Any]:
    try:
        return check_connection(Neo4jConfig.from_env())
    except Exception as exc:
        return {"status": "connection_failed", "error": f"{type(exc).__name__}: {exc}"}


def evidence_label(item: dict[str, Any], prefix: str, position: int) -> str:
    document = item.get("document_number") or item.get("document_id") or "không rõ văn bản"
    chunk = item.get("id") or item.get("element_id")
    return f"[{prefix}{position}] {document} · chunk {chunk}"


st.title("🔗 Multi-hop Graph RAG — Tra cứu pháp luật")
st.caption("MSMARCO tiếng Việt → Neo4j Vector Search → Quan hệ đa bước → Gemini")
status, qa_config = database_status(), QAConfig.from_env()

with st.sidebar:
    st.subheader("Cấu hình truy vấn")
    top_k = st.slider("Số seed vector (top-k)", 1, 20, 5)
    max_hops = st.slider("Số bước nhảy", 0, 5, 1)
    use_gemini = st.checkbox("Gọi Gemini để tạo câu trả lời", value=True)
    st.write(f"Neo4j: **{'Đã kết nối' if status.get('status') == 'connected' else 'Chưa kết nối'}**")
    if status.get("status") == "connected":
        st.write(f"Database: `{status['database']}` · {status['node_count']} node")
    else:
        st.error(status.get("error", "Không rõ lỗi"))
    st.write(f"Gemini key: **{'Đã cấu hình' if qa_config.api_key else 'Chưa cấu hình'}**")
    st.write(f"Generation model: `{qa_config.model}`")
    if st.button("Làm mới trạng thái"):
        database_status.clear()
        st.rerun()

question = st.text_area("Câu hỏi", placeholder="Ví dụ: Thông tư 41/2016/TT-NHNN căn cứ vào luật nào?", height=110)
if st.button("Tìm kiếm và trả lời", type="primary", disabled=not question.strip()):
    started = time.perf_counter()
    try:
        with st.spinner("Đang tìm kiếm vector và mở rộng đồ thị…"):
            retrieval = search_context(
                question, retrieval_config=RetrievalConfig(top_k=top_k, max_hops=max_hops),
                embedder=embedding_model(),
            )
        if use_gemini:
            with st.spinner("Đang tổng hợp câu trả lời bằng Gemini…"):
                result = answer_question(question, retrieval=retrieval, qa_config=qa_config)
        else:
            result = {"status": "retrieval_only", "answer": None, "retrieval": retrieval,
                      "generation_call_count": 0,
                      "warning": "Chế độ chỉ tìm kiếm: không gọi Gemini."}
        result["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    except Exception as exc:
        result = {"status": "error", "answer": None, "warning": f"{type(exc).__name__}: {exc}",
                  "retrieval": {"seeds": [], "related": []}, "generation_call_count": 0,
                  "total_latency_ms": round((time.perf_counter() - started) * 1000, 2)}
    st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader("Câu trả lời")
    if result.get("status") == "answered":
        st.success(result["answer"])
    elif result.get("status") == "insufficient_evidence":
        st.warning(result.get("answer"))
    elif result.get("status") == "retrieval_only":
        st.info(result.get("warning"))
    else:
        st.error(result.get("warning") or f"Trạng thái: {result.get('status')}")
    retrieval = result.get("retrieval", {})
    seeds, related = retrieval.get("seeds", []), retrieval.get("related", [])
    hierarchy, graph_facts = retrieval.get("hierarchy", []), retrieval.get("graph_facts", [])
    columns = st.columns(5)
    columns[0].metric("Seed", len(seeds)); columns[1].metric("Related", len(related))
    columns[2].metric("Hierarchy", len(hierarchy))
    columns[3].metric("Hop", retrieval.get("max_hops", max_hops))
    columns[4].metric("Latency", f"{result.get('total_latency_ms', 0):,.0f} ms")
    seed_tab, related_tab, hierarchy_tab, trace_tab = st.tabs(
        ["Seed vector", "Ngữ cảnh multi-hop", "Cùng Điều/Khoản", "Trace"]
    )
    with seed_tab:
        if not seeds: st.info("Không có seed.")
        for i, item in enumerate(seeds, 1):
            with st.expander(evidence_label(item, "S", i)):
                st.write(item.get("text") or "(không có nội dung)")
                st.caption(f"Vector score: {item.get('score')} · {item.get('document_title') or ''}")
    with related_tab:
        for fact in graph_facts:
            st.success(
                f"{fact.get('source_number')} —{fact.get('relationship')}→ {fact.get('target_number')}"
            )
        if not related: st.info("Không tìm thấy đoạn liên quan trong phạm vi hop đã chọn.")
        for i, item in enumerate(related, 1):
            with st.expander(evidence_label(item, "R", i)):
                st.write(item.get("text") or "(không có nội dung)")
                path = " → ".join(item.get("relationship_path", [])) or "—"
                st.caption(f"Hop: {item.get('hop')} · Quan hệ: {path} · Score: {item.get('related_score')}")
    with hierarchy_tab:
        if not hierarchy: st.info("Không có mục cùng Điều/Khoản.")
        for i, item in enumerate(hierarchy, 1):
            with st.expander(evidence_label(item, "H", i)):
                st.write(item.get("text") or "(không có nội dung)")
                st.caption(f"Thuộc: {item.get('parent_title') or 'không rõ'}")
    with trace_tab:
        st.json({"status": result.get("status"), "generation_call_count": result.get("generation_call_count", 0),
                 "model": result.get("model", qa_config.model), "top_k": retrieval.get("top_k", top_k),
                 "max_hops": retrieval.get("max_hops", max_hops), "relationships": retrieval.get("relationships", []),
                 "context_truncated": result.get("context_truncated", False)})

st.caption("Kết quả chỉ phục vụ tra cứu, không phải tư vấn pháp lý. Luôn kiểm tra lại văn bản gốc.")
