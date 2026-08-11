"""Giao diện Streamlit chỉ đọc để trực quan hóa output của Buổi 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "output"
RAW_DIR = OUTPUT_DIR / "raw"
CHUNK_DIR = OUTPUT_DIR / "chunks"
STRATEGIES = ("fixed-size", "semantic", "hierarchical")


@st.cache_data(show_spinner=False)
def load_json(path_string: str) -> Any:
    """Đọc JSON output; không đọc .env và không gọi dịch vụ bên ngoài."""
    with Path(path_string).open("r", encoding="utf-8") as file:
        return json.load(file)


def document_names() -> list[str]:
    names = {path.stem for path in RAW_DIR.glob("*.json")}
    names.update(path.stem.removesuffix("_chunks") for path in CHUNK_DIR.glob("*_chunks.json"))
    names.update(path.stem.removesuffix("_summary") for path in OUTPUT_DIR.glob("*_summary.json"))
    return sorted(names)


def read_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(str(path))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"Không thể đọc {path.name}: {type(error).__name__}.")
        return None


def summary_for(document: str) -> tuple[dict[str, Any], list[str]]:
    payload = read_optional_json(OUTPUT_DIR / f"{document}_summary.json")
    if not isinstance(payload, dict):
        return {}, []
    # Hỗ trợ cả schema cũ ({strategy: stats}) lẫn schema mới có summary/warnings.
    summary = payload.get("summary", payload)
    warnings = payload.get("warnings", [])
    return (summary if isinstance(summary, dict) else {}), warnings if isinstance(warnings, list) else []


def show_summary(document: str) -> None:
    summary, warnings = summary_for(document)
    st.subheader("So sánh chiến lược chunking")
    if not summary:
        st.info("Chưa có summary. Hãy chạy `process_pdf.py --write` để tạo output.")
        return

    rows = []
    for strategy in STRATEGIES:
        stats = summary.get(strategy, {})
        rows.append({
            "Chiến lược": strategy,
            "Số chunk": stats.get("count", 0),
            "Ngắn nhất": stats.get("min", 0),
            "Dài nhất": stats.get("max", 0),
            "Trung bình": stats.get("avg", 0.0),
        })
    columns = st.columns(3)
    for column, row in zip(columns, rows):
        column.metric(row["Chiến lược"], row["Số chunk"], "chunks")
    st.bar_chart({row["Chiến lược"]: row["Số chunk"] for row in rows})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    for warning in warnings:
        st.warning(warning)


def load_chunks(document: str) -> list[dict[str, Any]]:
    value = read_optional_json(CHUNK_DIR / f"{document}_chunks.json")
    return value if isinstance(value, list) else []


def show_chunks(document: str, strategy: str) -> None:
    chunks = [item for item in load_chunks(document) if item.get("strategy") == strategy]
    st.subheader(f"Chunks: {strategy}")
    if not chunks:
        st.info("Không có chunk phù hợp trong output hiện tại.")
        return

    pages = sorted({item.get("page_start") for item in chunks if isinstance(item.get("page_start"), int)})
    selected_pages = st.multiselect("Lọc theo trang", pages, default=pages, key=f"pages-{document}-{strategy}")
    query = st.text_input("Tìm trong nội dung chunk", placeholder="Ví dụ: tổ chức tín dụng", key=f"query-{document}-{strategy}").casefold().strip()
    filtered = [
        item for item in chunks
        if item.get("page_start") in selected_pages and (not query or query in str(item.get("text", "")).casefold())
    ]
    st.caption(f"Hiển thị {len(filtered)}/{len(chunks)} chunk.")
    maximum = min(len(filtered), 100)
    amount = st.slider("Số chunk cần xem", 1, maximum, min(10, maximum), key=f"amount-{document}-{strategy}") if maximum else 0

    for item in filtered[:amount]:
        title = f"{item.get('chunk_id', 'chunk')} — trang {item.get('page_start', '?')}–{item.get('page_end', '?')}"
        with st.expander(title):
            st.caption(f"Độ dài: {len(str(item.get('text', '')))} ký tự")
            st.json(item.get("metadata", {}))
            st.code(str(item.get("text", "")), language=None)


def show_raw_pages(document: str) -> None:
    pages = read_optional_json(RAW_DIR / f"{document}.json")
    if not isinstance(pages, list):
        st.info("Không có raw JSON cho tài liệu này.")
        return
    st.subheader("Raw text theo trang")
    for page in pages:
        title = f"Trang {page.get('page', '?')} — OCR: {'có' if page.get('ocr_used') else 'không'}"
        with st.expander(title):
            st.caption(f"Nguồn: {page.get('source', '?')} | Ngôn ngữ: {page.get('language', '?')}")
            st.json(page.get("metadata", {}))
            st.code(str(page.get("text", "")), language=None)


def main() -> None:
    st.set_page_config(page_title="RAG Buổi 5", page_icon="📄", layout="wide")
    st.title("📄 Trực quan hóa chunk RAG — Buổi 5")
    st.caption("Giao diện chỉ đọc `output/`: không sửa PDF, không đọc secret, không gọi API hoặc LLM.")

    documents = document_names()
    if not documents:
        st.warning("Chưa tìm thấy JSON trong output. Hãy chạy `process_pdf.py --write` trước.")
        return

    with st.sidebar:
        st.header("Bộ lọc")
        document = st.selectbox("Tài liệu", documents)
        strategy = st.selectbox("Chiến lược", STRATEGIES)
        show_raw = st.checkbox("Hiển thị raw text theo trang")
        st.divider()
        st.caption("Dữ liệu chỉ được đọc từ thư mục output.")

    show_summary(document)
    show_chunks(document, strategy)
    if show_raw:
        show_raw_pages(document)


if __name__ == "__main__":
    main()
