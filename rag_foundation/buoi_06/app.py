"""Simple Streamlit interface for the workshop RAG."""

import streamlit as st

import rag


st.set_page_config(page_title="RAG Workshop", page_icon="🔎")
st.title("RAG Workshop")
st.caption("Question ➜ Top-k ➜ Gemini ➜ Answer")


def postgres_status():
    connection = rag._postgres_connection()
    if connection is None:
        return "Không kết nối (sẽ dùng SQLite khi index)"
    connection.close()
    return "Sẵn sàng"


def chroma_status():
    try:
        rag._chroma_collection()
        return rag.CHROMA_MODE
    except Exception:
        return "Không sẵn sàng"


with st.sidebar:
    st.header("Trạng thái")
    st.write(f"**PostgreSQL:** {postgres_status()}")
    st.write(f"**ChromaDB:** {chroma_status()}")
    st.write(
        "**Gemini API Key:** "
        + ("Có" if rag._gemini_client() else "Thiếu — chỉ Retrieval")
    )


if st.button("Index dữ liệu", type="primary"):
    try:
        with st.spinner("Đang tạo index..."):
            result = rag.index()
        st.success(f"Đã index {result['documents']} document và {result['chunks']} chunk.")
    except Exception:
        st.error("Không thể index dữ liệu. Hãy kiểm tra kết nối Gemini và storage.")

question = st.text_input("Câu hỏi")
k = st.number_input("Top-k", min_value=1, max_value=10, value=3, step=1)

if st.button("Hỏi"):
    if not question.strip():
        st.warning("Hãy nhập câu hỏi.")
    else:
        try:
            with st.spinner("Đang tìm câu trả lời..."):
                result = rag.ask(question, k=int(k), include_context=True)
        except Exception:
            st.error("Không thể xử lý câu hỏi. Hãy kiểm tra index và kết nối Gemini.")
        else:
            st.subheader("Kết quả Top-k")
            if result["chunks"]:
                for number, text in enumerate(result["chunks"], start=1):
                    with st.expander(f"Chunk {number}"):
                        st.write(text)
            else:
                st.info("Không có chunk để hiển thị.")

            st.subheader("Answer")
            if result["answer"]:
                st.write(result["answer"])
            elif result["chunks"]:
                st.info("Thiếu Gemini API Key nên chỉ hiển thị Retrieval.")
