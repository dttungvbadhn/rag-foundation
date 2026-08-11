"""Giao diện Streamlit tối giản cho pipeline RAG Buổi 07."""

import streamlit as st

from rag import RagIndexError, answer_question, get_index_status, index_chunks, load_config


def render_evidence(evidence: list[dict]) -> None:
    st.subheader("Nguồn tham khảo")
    st.caption(
        "Cosine distance thấp hơn thường cho thấy nội dung liên quan hơn; "
        "distance không phải là xác suất."
    )
    if not evidence:
        st.info("Chưa có evidence.")
        return

    for item in evidence:
        page = (
            str(item["page_start"])
            if item["page_start"] == item["page_end"]
            else f"{item['page_start']}-{item['page_end']}"
        )
        summary = f"{item['source']} – tr. {page} – {item['chunk_id']}"
        if not item["accepted"]:
            summary += " – Không đạt confidence gate"
        with st.expander(summary):
            st.write(f"**Evidence ID:** {item['evidence_id']}")
            st.write(f"**Source:** {item['source']}")
            st.write(f"**Trang:** {page}")
            st.write(f"**Chunk ID:** {item['chunk_id']}")
            st.write(f"**Distance:** {item['distance']:.6f}")
            if item["accepted"]:
                st.success("Accepted: Đạt confidence gate và có thể được dùng để tạo answer.")
            else:
                st.warning("Accepted: Không đạt confidence gate; không được dùng để tạo answer.")
            st.write(item["text"])


st.set_page_config(page_title="RAG Buổi 07", page_icon="📚", layout="wide")
st.title("RAG Buổi 07")
st.caption("Semantic retrieval với Gemini embedding và ChromaDB persistent")

try:
    config = load_config()
except RagIndexError as exc:
    st.error(f"Không thể đọc cấu hình: {exc}")
    st.stop()
except Exception:
    st.error("Không thể đọc cấu hình. Hãy kiểm tra file .env.")
    st.stop()

strategy = st.sidebar.selectbox(
    "Strategy",
    options=["hierarchical", "semantic", "fixed-size"],
)
top_k = st.sidebar.slider(
    "Top-k",
    min_value=1,
    max_value=10,
    value=min(max(config.default_top_k, 1), 10),
)

try:
    status = get_index_status(config, strategy)
    status_error = None
except RagIndexError as exc:
    status = {
        "collection_name": "Không xác định",
        "collection_exists": False,
        "record_count": 0,
    }
    status_error = str(exc)
except Exception:
    status = {
        "collection_name": "Không xác định",
        "collection_exists": False,
        "record_count": 0,
    }
    status_error = "Không thể đọc trạng thái collection."

st.sidebar.subheader("Trạng thái hệ thống")
st.sidebar.write(f"**API key:** {'Có' if config.api_key else 'Thiếu'}")
st.sidebar.write(f"**Embedding model:** {config.embedding_model}")
st.sidebar.write(f"**Embedding dimension:** {config.embedding_dim}")
st.sidebar.write(f"**Generation model:** {config.generation_model}")
st.sidebar.write(f"**Strategy:** {strategy}")
st.sidebar.write(f"**Collection:** {status['collection_name']}")
st.sidebar.write(
    f"**Collection tồn tại:** {'Có' if status['collection_exists'] else 'Không'}"
)
st.sidebar.write(f"**Số chunk:** {status['record_count']}")
st.sidebar.write(f"**RAG_MAX_DISTANCE:** {config.max_distance}")
if status_error:
    st.sidebar.warning(status_error)

st.header("Index dữ liệu")
reset = st.checkbox("Reset collection trước khi index")
if st.button("Index dữ liệu", type="primary"):
    if not config.api_key:
        st.error("Thiếu API key. Hãy điền GEMINI_API_KEY trong file .env rồi chạy lại.")
    else:
        before_count = status["record_count"]
        try:
            with st.spinner("Đang tạo embedding và index dữ liệu..."):
                indexed = index_chunks(config, strategy, reset=reset)
            st.session_state["last_index_result"] = {
                **indexed,
                "strategy": strategy,
                "before_count": before_count,
            }
            st.rerun()
        except RagIndexError as exc:
            st.error(f"Index thất bại: {exc}")
        except Exception:
            st.error("Index thất bại do lỗi không mong đợi. Không có stack trace được hiển thị.")

last_index = st.session_state.get("last_index_result")
if last_index:
    loader_stats = last_index["loader_stats"]
    st.success("Lần index gần nhất đã hoàn thành.")
    st.write(f"**Strategy:** {last_index['strategy']}")
    st.write(f"**Collection:** {last_index['collection_name']}")
    st.write(f"**Số chunk trước:** {last_index['before_count']}")
    st.write(f"**Số chunk sau:** {last_index['record_count']}")
    st.write(f"**Text rỗng bị bỏ qua:** {loader_stats['empty_text_skipped']}")

st.divider()
st.header("Hỏi đáp")
question = st.text_area(
    "Nhập câu hỏi",
    max_chars=2000,
    placeholder="Ví dụ: Nội dung chính của tài liệu là gì?",
)
if st.button("Gửi câu hỏi", type="primary"):
    if not question.strip():
        st.warning("Vui lòng nhập câu hỏi trước khi gửi.")
    elif not config.api_key:
        st.error("Thiếu API key. Hãy điền GEMINI_API_KEY trong file .env rồi chạy lại.")
    elif not status["collection_exists"]:
        st.warning("Collection của strategy này chưa tồn tại. Hãy index dữ liệu trước.")
    elif status["record_count"] < 1:
        st.warning("Collection đang rỗng. Hãy index dữ liệu trước.")
    else:
        try:
            with st.spinner("Đang retrieval và tạo câu trả lời..."):
                st.session_state["last_query_result"] = answer_question(
                    question=question,
                    top_k=top_k,
                    strategy=strategy,
                    config=config,
                )
        except RagIndexError as exc:
            st.error(f"Không thể xử lý câu hỏi: {exc}")
        except Exception:
            st.error("Không thể xử lý câu hỏi do lỗi không mong đợi.")

result = st.session_state.get("last_query_result")
if result:
    st.subheader("Kết quả")
    result_status = result["status"]
    if result_status == "answered":
        st.success("Status: answered")
        st.write(result["answer"])
    elif result_status == "insufficient_evidence":
        st.warning("Status: insufficient_evidence — không tìm thấy đủ thông tin liên quan.")
        st.info(result["answer"])
    elif result_status == "retrieval_only":
        st.warning("Status: retrieval_only — đã retrieve được nguồn nhưng generation lỗi.")
        st.info(result["answer"])
    else:
        st.warning("Status không xác định.")

    for warning in result.get("warnings", []):
        st.warning(warning)

    citations = result.get("citations", [])
    if citations:
        st.subheader("Citation")
        for citation in citations:
            st.write(citation["display"])

    render_evidence(result.get("evidence", []))
else:
    render_evidence([])
