"""Build an Obsidian-compatible Markdown wiki from normalized graph CSVs."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
TYPE_DIR = {"RuiRo": "risks", "KiemSoat": "controls", "SuKienRuiRo": "events"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", value).strip().rstrip(".")
    return value or "untitled"


def yaml_value(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def link(entity: dict[str, str]) -> str:
    target = f"{TYPE_DIR[entity['type']]}/{safe_filename(entity['name'])}"
    return f"[[{target}|{entity['name']}]]"


def detail(label: str, value: str) -> str:
    return f"- **{label}:** {value}" if value else f"- **{label}:** Chưa có dữ liệu."


def relation_block(relation: dict[str, str], other: dict[str, str]) -> str:
    return "\n".join([
        f"- {link(other)}",
        f"  - relationship_type: `{relation['relationship_type']}`",
        f"  - evidence_quote: {relation['evidence_quote'] or 'Chưa có dữ liệu.'}",
        f"  - verification_status: `{relation['verification_status'] or 'Chưa có dữ liệu.'}`",
    ])


def main() -> None:
    entities = read_rows(ROOT / "outputs" / "entities.csv")
    relations = read_rows(ROOT / "outputs" / "relations.csv")
    by_id = {row["id"]: row for row in entities}
    outgoing: dict[str, list[dict[str, str]]] = defaultdict(list)
    incoming: dict[str, list[dict[str, str]]] = defaultdict(list)
    for relation in relations:
        outgoing[relation["source_id"]].append(relation)
        incoming[relation["target_id"]].append(relation)

    for folder in TYPE_DIR.values():
        (WIKI / folder).mkdir(parents=True, exist_ok=True)

    page_count = 0
    link_count = 0
    for entity in entities:
        lines = [
            "---", f"id: {yaml_value(entity['id'])}", f"type: {yaml_value(entity['type'])}",
            f"verification_status: {yaml_value(entity.get('verification_status', ''))}",
            f"data_origin: {yaml_value(entity.get('data_origin', ''))}", "---", "", f"# {entity['name']}", "",
            detail("ID", entity["id"]), detail("Mô tả", entity.get("description", "")),
        ]
        if entity["type"] == "RuiRo":
            fields = [("Phân loại", "category"), ("Nguyên nhân", "cause"), ("Sự kiện", "event"),
                      ("Tác động", "impact"), ("Mức rủi ro vốn có", "inherent_level"),
                      ("Mức rủi ro còn lại", "residual_level"), ("Mã đơn vị sở hữu", "owner_unit_id")]
            lines.extend(detail(label, entity.get(key, "")) for label, key in fields)
            controls = [r for r in incoming[entity["id"]] if r["relationship_type"] == "MITIGATES"]
            events = [r for r in outgoing[entity["id"]] if r["relationship_type"] == "OBSERVED_AS"]
            lines += ["", "## Kiểm soát liên quan", ""]
            lines += [relation_block(r, by_id[r["source_id"]]) for r in controls] or ["Chưa có dữ liệu."]
            lines += ["", "## Sự kiện rủi ro liên quan", ""]
            lines += [relation_block(r, by_id[r["target_id"]]) for r in events] or ["Chưa có dữ liệu."]
            link_count += len(controls) + len(events)
        elif entity["type"] == "KiemSoat":
            fields = [("Loại kiểm soát", "control_type"), ("Tần suất", "frequency"),
                      ("Mã vai trò phụ trách", "owner_role_id"), ("Hiệu lực", "effectiveness")]
            lines.extend(detail(label, entity.get(key, "")) for label, key in fields)
            related = [r for r in outgoing[entity["id"]] if r["relationship_type"] == "MITIGATES"]
            lines += ["", "## Rủi ro được giảm thiểu", ""]
            lines += [relation_block(r, by_id[r["target_id"]]) for r in related] or ["Chưa có dữ liệu."]
            link_count += len(related)
        else:
            fields = [("Rủi ro tham chiếu", "risk_id"), ("Ngày xảy ra", "occurred_at"),
                      ("Ngày phát hiện", "discovered_at"), ("Mức độ", "severity"),
                      ("Tổn thất (VND)", "loss_amount_vnd")]
            lines.extend(detail(label, entity.get(key, "")) for label, key in fields)
            related = [r for r in incoming[entity["id"]] if r["relationship_type"] == "OBSERVED_AS"]
            lines += ["", "## Rủi ro liên quan", ""]
            lines += [relation_block(r, by_id[r["source_id"]]) for r in related] or ["Chưa có dữ liệu."]
            link_count += len(related)
        path = WIKI / TYPE_DIR[entity["type"]] / f"{safe_filename(entity['name'])}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        page_count += 1

    counts = Counter(row["type"] for row in entities)
    home = ["# Wiki Risk Graph", "", "Wiki tri thức rủi ro được sinh từ dữ liệu CSV nguồn.", "",
            "## Danh mục", ""]
    for entity_type, title in [("RuiRo", "Rủi ro"), ("KiemSoat", "Kiểm soát"), ("SuKienRuiRo", "Sự kiện rủi ro")]:
        home += [f"### {title}", ""] + [f"- {link(row)}" for row in entities if row["type"] == entity_type] + [""]
    home += ["## Thống kê", "", f"- Tổng node: {len(entities)}", f"- Tổng edge: {len(relations)}",
             f"- RuiRo: {counts['RuiRo']}", f"- KiemSoat: {counts['KiemSoat']}",
             f"- SuKienRuiRo: {counts['SuKienRuiRo']}"]
    (WIKI / "Home.md").write_text("\n".join(home) + "\n", encoding="utf-8")
    link_count += len(entities)
    print(f"Created {page_count + 1} Markdown pages with {link_count} wikilinks")


if __name__ == "__main__":
    main()
