from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_logger import read_events
from scripts.compliance_gap import assess
from scripts.internal_lookup import lookup
from scripts.rbac import VALID_ROLES

st.set_page_config(page_title="Secure RAG & Compliance — Buổi 17", layout="wide")
st.title("SECURE RAG & COMPLIANCE — BUỔI 17")
st.warning("Demo đào tạo — kết quả AI cần kiểm toán viên xác minh.")
user_id = st.sidebar.text_input("User ID demo", "demo01")
role = st.sidebar.selectbox("User Role", VALID_ROLES)
st.sidebar.caption("Neo4j: không dùng cho gap matching (không có edge phù hợp đã được xác minh).")

lookup_tab, gap_tab, audit_tab = st.tabs(["TRA CỨU QUY ĐỊNH", "COMPLIANCE GAP CHECKER", "AUDIT"])
with lookup_tab:
    question = st.text_area("Câu hỏi")
    top_k = st.slider("Top-k", 1, 10, 5)
    if st.button("Tra cứu") and question.strip():
        result = lookup(question, role, top_k, user_id)
        st.write(result["answer"]); st.caption(f"Request ID: {result['request_id']}")
        st.write("Access decision:", result["access_decision"])
        if result["access_decision"] == "ALLOW":
            st.dataframe([{k: row[k] for k in ("rank", "document_id", "chunk_id", "citation")} for row in result["results"]])
with gap_tab:
    requirement = st.text_area("Yêu cầu NHNN")
    if st.button("Kiểm tra gap") and requirement.strip():
        finding = assess(requirement); st.dataframe([finding]); st.warning("NEEDS_HUMAN_REVIEW")
with audit_tab:
    events = read_events()
    visible = events if role == "Admin" else [event for event in events if event.get("user_role") == role]
    st.dataframe(visible)

