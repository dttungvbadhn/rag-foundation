"""Streamlit explorer for the generated Wiki Risk Graph."""

from __future__ import annotations

import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
TYPE_LABEL = {"RuiRo": "Rủi ro", "KiemSoat": "Kiểm soát", "SuKienRuiRo": "Sự kiện rủi ro"}
TYPE_COLOR = {"RuiRo": "#ef4444", "KiemSoat": "#22c55e", "SuKienRuiRo": "#3b82f6"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@st.cache_data(show_spinner=False)
def load_graph(entity_mtime: float, relation_mtime: float) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    del entity_mtime, relation_mtime
    return read_csv(OUTPUTS / "entities.csv"), read_csv(OUTPUTS / "relations.csv")


def graphviz_source(nodes: list[dict[str, str]], relations: list[dict[str, str]]) -> str:
    ids = {node["id"] for node in nodes}
    lines = ["digraph RiskGraph {", 'rankdir="LR";', 'graph [bgcolor="transparent", pad="0.3"];',
             'node [shape="box", style="rounded,filled", fontname="Arial", fontcolor="white"];',
             'edge [fontname="Arial", fontsize="9", color="#64748b"];']
    for node in nodes:
        label = f"{node['id']}\\n{node['name']}".replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'"{node["id"]}" [label="{label}", fillcolor="{TYPE_COLOR[node["type"]]}"];')
    for relation in relations:
        if relation["source_id"] in ids and relation["target_id"] in ids:
            lines.append(f'"{relation["source_id"]}" -> "{relation["target_id"]}" [label="{relation["relationship_type"]}"];')
    lines.append("}")
    return "\n".join(lines)


def relation_rows(entity_id: str, entities_by_id: dict[str, dict[str, str]], relations: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for relation in relations:
        if entity_id not in (relation["source_id"], relation["target_id"]):
            continue
        other_id = relation["target_id"] if relation["source_id"] == entity_id else relation["source_id"]
        result.append({
            "Hướng": "Đi ra" if relation["source_id"] == entity_id else "Đi vào",
            "Quan hệ": relation["relationship_type"],
            "Node liên quan": entities_by_id.get(other_id, {}).get("name", other_id),
            "ID": other_id,
            "Bằng chứng": relation["evidence_quote"],
            "Xác minh": relation["verification_status"],
            "Nguồn": relation["source"],
        })
    return result


st.set_page_config(page_title="Wiki Risk Graph", page_icon="🕸️", layout="wide")
st.title("Wiki Risk Graph")
st.caption("Tra cứu KiemSoat → RuiRo → SuKienRuiRo từ dữ liệu mô phỏng")

entity_path = OUTPUTS / "entities.csv"
relation_path = OUTPUTS / "relations.csv"
if not entity_path.exists() or not relation_path.exists():
    st.error("Chưa có output. Hãy chạy `python scripts/build_entities.py` trước.")
    st.stop()

entities, relations = load_graph(entity_path.stat().st_mtime, relation_path.stat().st_mtime)
by_id = {row["id"]: row for row in entities}
counts = Counter(row["type"] for row in entities)

with st.sidebar:
    st.header("Bộ lọc")
    selected_types = st.multiselect(
        "Loại node", list(TYPE_LABEL), default=list(TYPE_LABEL), format_func=lambda value: TYPE_LABEL[value]
    )
    statuses = sorted({row["verification_status"] for row in entities} | {row["verification_status"] for row in relations})
    selected_statuses = st.multiselect("Trạng thái xác minh", statuses, default=statuses)
    query = st.text_input("Tìm theo ID hoặc nội dung", placeholder="Ví dụ: RR-001, giao dịch...").strip().casefold()
    st.divider()
    st.caption("Màu node")
    for entity_type, label in TYPE_LABEL.items():
        st.markdown(f'<span style="color:{TYPE_COLOR[entity_type]}">●</span> {html.escape(label)}', unsafe_allow_html=True)

filtered = [
    row for row in entities
    if row["type"] in selected_types
    and row["verification_status"] in selected_statuses
    and (not query or query in " ".join(row.values()).casefold())
]
filtered_ids = {row["id"] for row in filtered}
filtered_relations = [
    row for row in relations
    if row["source_id"] in filtered_ids and row["target_id"] in filtered_ids
    and row["verification_status"] in selected_statuses
]

metric_columns = st.columns(5)
metric_columns[0].metric("Tổng node", len(entities))
metric_columns[1].metric("Rủi ro", counts["RuiRo"])
metric_columns[2].metric("Kiểm soát", counts["KiemSoat"])
metric_columns[3].metric("Sự kiện", counts["SuKienRuiRo"])
metric_columns[4].metric("Quan hệ", len(relations))

tab_graph, tab_lookup, tab_quality, tab_data = st.tabs(["Đồ thị", "Tra cứu hồ sơ", "Chất lượng dữ liệu", "Dữ liệu"])

with tab_graph:
    st.subheader(f"Đồ thị đang hiển thị: {len(filtered)} node, {len(filtered_relations)} edge")
    if filtered:
        st.graphviz_chart(graphviz_source(filtered, filtered_relations), width="stretch")
    else:
        st.info("Không có node phù hợp với bộ lọc.")

with tab_lookup:
    options = sorted(filtered, key=lambda row: (row["type"], row["id"]))
    if not options:
        st.info("Không có hồ sơ phù hợp với bộ lọc.")
    else:
        selected_id = st.selectbox(
            "Chọn hồ sơ", [row["id"] for row in options],
            format_func=lambda value: f"{value} — {by_id[value]['name']}"
        )
        entity = by_id[selected_id]
        st.markdown(f"### {entity['name']}")
        st.caption(f"{TYPE_LABEL[entity['type']]} · {entity['id']} · {entity['verification_status']}")
        hidden = {"id", "type", "name", "source_file"}
        details = [{"Thuộc tính": key, "Giá trị": value or "Chưa có dữ liệu."} for key, value in entity.items() if key not in hidden]
        st.dataframe(details, width="stretch", hide_index=True)
        st.markdown("#### Quan hệ và bằng chứng")
        related = relation_rows(selected_id, by_id, relations)
        if related:
            st.dataframe(related, width="stretch", hide_index=True)
        else:
            st.warning("Hồ sơ chưa có quan hệ trong dữ liệu nguồn.")

with tab_quality:
    mitigated = {row["target_id"] for row in relations if row["relationship_type"] == "MITIGATES"}
    observed = {row["source_id"] for row in relations if row["relationship_type"] == "OBSERVED_AS"}
    risks = [row for row in entities if row["type"] == "RuiRo"]
    no_controls = [row for row in risks if row["id"] not in mitigated]
    no_events = [row for row in risks if row["id"] not in observed]
    q1, q2 = st.columns(2)
    q1.metric("Rủi ro chưa có kiểm soát", len(no_controls))
    q2.metric("Rủi ro chưa có sự kiện", len(no_events))
    if no_controls:
        st.warning("Rủi ro chưa có kiểm soát (không tự động bịa quan hệ)")
        st.dataframe([{"ID": row["id"], "Tên": row["name"]} for row in no_controls], hide_index=True, width="stretch")
    report_path = OUTPUTS / "wiki_validation_report.md"
    if report_path.exists():
        with st.expander("Xem báo cáo validation đầy đủ"):
            st.markdown(report_path.read_text(encoding="utf-8"))

with tab_data:
    st.markdown("#### Entities")
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button("Tải entities.csv", entity_path.read_bytes(), "entities.csv", "text/csv")
    st.markdown("#### Relations")
    st.dataframe(filtered_relations, width="stretch", hide_index=True)
    st.download_button("Tải relations.csv", relation_path.read_bytes(), "relations.csv", "text/csv")
