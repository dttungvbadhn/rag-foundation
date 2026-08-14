from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app_backend import (
    RELATIONSHIP_TYPES,
    Neo4jRepository,
    csv_statistics,
    document_bundle,
    dot_graph,
    load_app_data,
    search_documents,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
load_dotenv(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="Vietnamese Legal Knowledge Graph",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; padding:14px; border-radius:12px;}
      .doc-card {padding:1rem; border:1px solid #e2e8f0; border-radius:12px; background:#fff;}
      .small-muted {color:#64748b; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_app_data(BASE_DIR)


@st.cache_resource(show_spinner=False)
def get_neo4j() -> Neo4jRepository | None:
    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    if not all((uri, user, password)):
        return None
    return Neo4jRepository(uri, user, password, database)


def neo4j_status() -> tuple[Neo4jRepository | None, str]:
    repository = get_neo4j()
    if repository is None:
        return None, "Chưa cấu hình"
    try:
        return (repository, "Đã kết nối") if repository.verify() else (None, "Lỗi kết nối")
    except Exception:
        return None, "Lỗi kết nối"


def show_overview(data, neo4j: Neo4jRepository | None) -> None:
    st.header("Tổng quan Knowledge Graph")
    stats = csv_statistics(data)
    columns = st.columns(4)
    columns[0].metric("Văn bản", stats["documents"])
    columns[1].metric("Entity canonical", stats["entities"])
    columns[2].metric("Relationship", stats["relationships"])
    columns[3].metric("Validation PASS", stats["validation"].get("PASS", 0))

    left, right = st.columns(2)
    with left:
        st.subheader("Entity theo loại")
        entity_chart = pd.DataFrame.from_dict(
            stats["entity_types"], orient="index", columns=["Số lượng"]
        )
        st.bar_chart(entity_chart, horizontal=True)
    with right:
        st.subheader("Relationship theo loại")
        relationship_chart = pd.DataFrame.from_dict(
            stats["relationship_types"], orient="index", columns=["Số lượng"]
        )
        st.bar_chart(relationship_chart, horizontal=True)

    if neo4j:
        graph_stats = neo4j.statistics()
        csv_nodes = {"Document": stats["documents"], **stats["entity_types"]}
        nodes_match = graph_stats["nodes"] == csv_nodes
        relationships_match = graph_stats["relationships"] == stats["relationship_types"]
        st.subheader("Đối chiếu Neo4j")
        c1, c2 = st.columns(2)
        c1.success("Node khớp CSV" if nodes_match else "Node lệch CSV")
        c2.success("Relationship khớp CSV" if relationships_match else "Relationship lệch CSV")
        with st.expander("Chi tiết số liệu Neo4j"):
            st.json(graph_stats)


def show_document_detail(data, document_id: str) -> None:
    bundle = document_bundle(data, document_id)
    document = bundle["document"]
    st.subheader(f"{document['so_ky_hieu']} — {document['title']}")
    metadata_fields = [
        "loai_van_ban", "ngay_ban_hanh", "ngay_co_hieu_luc", "ngay_het_hieu_luc",
        "co_quan_ban_hanh", "nguoi_ky", "linh_vuc", "tinh_trang_hieu_luc",
    ]
    metadata = pd.DataFrame(
        {"Trường": metadata_fields, "Giá trị": [document.get(field, "") for field in metadata_fields]}
    )
    st.dataframe(metadata, hide_index=True, width="stretch")

    entity_tab, relation_tab, content_tab = st.tabs(["Entity & evidence", "Quan hệ", "Nội dung sạch"])
    with entity_tab:
        entities = bundle["entities"]
        if entities.empty:
            st.info("Không có entity.")
        else:
            st.dataframe(
                entities[["entity_type", "canonical_name", "original_name", "method", "confidence", "evidence"]],
                hide_index=True,
                width="stretch",
                column_config={"confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0)},
            )
    with relation_tab:
        relations = bundle["relationships"]
        if relations.empty:
            st.info("Không có relationship.")
        else:
            st.dataframe(
                relations[["source_name", "relationship_type", "target_name", "method", "confidence", "evidence"]],
                hide_index=True,
                width="stretch",
            )
    with content_tab:
        st.text_area("content_clean", value=str(document["content_clean"]), height=500, disabled=True)


def show_search(data) -> None:
    st.header("Tìm kiếm văn bản pháp luật")
    query = st.text_input("Từ khóa, số hiệu hoặc nội dung", placeholder="Ví dụ: 22/2023/TT-NHNN hoặc tỷ lệ an toàn vốn")
    col1, col2 = st.columns(2)
    types = sorted(data.documents["loai_van_ban"].dropna().unique())
    fields = sorted(data.documents["linh_vuc"].fillna("Chưa phân loại").unique())
    selected_types = col1.multiselect("Loại văn bản", types)
    selected_fields = col2.multiselect("Lĩnh vực", fields)
    result = search_documents(data.documents, query, selected_types, selected_fields)
    st.caption(f"Tìm thấy {len(result)} văn bản")
    st.dataframe(
        result[["id", "so_ky_hieu", "title", "ngay_ban_hanh", "linh_vuc"]],
        hide_index=True,
        width="stretch",
    )
    if result.empty:
        return
    options = result["id"].astype(str).tolist()
    labels = result.set_index(result["id"].astype(str))["so_ky_hieu"].astype(str).to_dict()
    selected = st.selectbox("Mở chi tiết văn bản", options, format_func=lambda item: labels[item])
    show_document_detail(data, selected)


def show_graph(data, neo4j: Neo4jRepository | None) -> None:
    st.header("Khám phá Knowledge Graph")
    if neo4j is None:
        st.error("Không thể kết nối Neo4j. Kiểm tra cấu hình `.env`.")
        return
    col1, col2 = st.columns([2, 1])
    selected_types = col1.multiselect(
        "Loại relationship", RELATIONSHIP_TYPES, default=["THAM_CHIEU", "SUA_DOI_BO_SUNG", "KY_BOI", "AP_DUNG_CHO"]
    )
    limit = col2.slider("Số edge tối đa", 10, 200, 80, 10)
    mode = st.radio("Phạm vi", ["Toàn graph", "Theo văn bản"], horizontal=True)
    document_id = None
    hops = 1
    if mode == "Theo văn bản":
        document_options = data.documents["id"].astype(str).tolist()
        labels = data.documents.set_index(data.documents["id"].astype(str))["so_ky_hieu"].astype(str).to_dict()
        document_id = st.selectbox("Văn bản trung tâm", document_options, format_func=lambda item: labels[item])
        hops = st.slider("Số hop", 1, 3, 1)
    try:
        nodes, edges = neo4j.graph(selected_types, limit, document_id, hops)
    except Exception as exc:
        st.error(f"Không thể truy vấn graph: {type(exc).__name__}")
        return
    if not edges:
        st.info("Không có edge phù hợp với bộ lọc.")
        return
    st.caption(f"{len(nodes)} node · {len(edges)} edge")
    st.graphviz_chart(dot_graph(nodes, edges), width="stretch")
    with st.expander("Bảng edge và evidence"):
        st.dataframe(pd.DataFrame(edges), hide_index=True, width="stretch")


def show_quality(data) -> None:
    st.header("Kiểm tra chất lượng và truy vết")
    status_counts = data.validation["status"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Relation raw", len(data.validation))
    c2.metric("PASS", int(status_counts.get("PASS", 0)))
    c3.metric("FAIL đã loại", int(status_counts.get("FAIL", 0)))
    failures = data.validation[data.validation["status"].eq("FAIL")]
    st.subheader("Relationship bị loại")
    st.dataframe(failures, hide_index=True, width="stretch")
    st.download_button(
        "Tải validation report",
        data=data.validation.to_csv(index=False).encode("utf-8-sig"),
        file_name="validation_report.csv",
        mime="text/csv",
    )


data = get_data()
neo4j, connection_status = neo4j_status()
with st.sidebar:
    st.title("⚖️ Legal Knowledge Graph")
    st.caption("30 văn bản pháp luật Việt Nam")
    page = st.radio(
        "Điều hướng",
        ["Tổng quan", "Tìm kiếm văn bản", "Khám phá graph", "Chất lượng dữ liệu"],
    )
    st.divider()
    if connection_status == "Đã kết nối":
        st.success(f"Neo4j: {connection_status}")
    else:
        st.warning(f"Neo4j: {connection_status}")
    st.caption(f"Database: {os.getenv('NEO4J_DATABASE', 'neo4j')}")

if page == "Tổng quan":
    show_overview(data, neo4j)
elif page == "Tìm kiếm văn bản":
    show_search(data)
elif page == "Khám phá graph":
    show_graph(data, neo4j)
else:
    show_quality(data)
