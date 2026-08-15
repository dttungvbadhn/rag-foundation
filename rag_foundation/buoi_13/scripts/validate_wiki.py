"""Validate generated Wiki pages, graph references, and Obsidian wikilinks."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIKI = ROOT / "wiki"
OUTPUT = ROOT / "outputs" / "wiki_validation_report.md"
LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
ID_RE = re.compile(r'^id:\s*["\']?([^"\'\n]+)', re.MULTILINE)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "outputs" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def display(items: list[str]) -> str:
    return "Không có." if not items else "\n".join(f"- `{item}`" for item in items)


def main() -> None:
    entities = read_csv("entities.csv")
    relations = read_csv("relations.csv")
    pages = sorted(WIKI.rglob("*.md"))
    entity_ids = [row["id"] for row in entities]
    known_ids = set(entity_ids)
    page_targets = {page.relative_to(WIKI).with_suffix("").as_posix() for page in pages}
    page_targets |= {Path(target).name for target in page_targets}
    page_ids: list[str] = []
    broken: list[str] = []
    orphan_pages: list[str] = []
    total_links = 0

    for page in pages:
        text = page.read_text(encoding="utf-8")
        ids = ID_RE.findall(text)
        page_ids.extend(value.strip() for value in ids)
        links = LINK_RE.findall(text)
        total_links += len(links)
        for target in links:
            normalized = target.strip().removesuffix(".md")
            if normalized not in page_targets and Path(normalized).name not in page_targets:
                broken.append(f"{page.relative_to(WIKI)} -> {target}")
        if page.name != "Home.md" and not links:
            orphan_pages.append(str(page.relative_to(WIKI)))

    duplicates = sorted(key for key, count in Counter(entity_ids).items() if count > 1)
    unknown_pages = sorted(set(page_ids) - known_ids)
    relation_orphans = sorted(
        f"{r['source_id']} -{r['relationship_type']}-> {r['target_id']}"
        for r in relations if r["source_id"] not in known_ids or r["target_id"] not in known_ids
    )
    mitigated: dict[str, int] = defaultdict(int)
    observed: dict[str, int] = defaultdict(int)
    for relation in relations:
        if relation["relationship_type"] == "MITIGATES":
            mitigated[relation["target_id"]] += 1
        elif relation["relationship_type"] == "OBSERVED_AS":
            observed[relation["source_id"]] += 1
    risks = [row["id"] for row in entities if row["type"] == "RuiRo"]
    no_controls = sorted(risk for risk in risks if not mitigated[risk])
    no_events = sorted(risk for risk in risks if not observed[risk])
    program_errors = broken + duplicates + unknown_pages + relation_orphans
    report = f"""# Báo cáo kiểm tra Wiki Risk Graph

## Tổng quan

- Tổng file Markdown: {len(pages)}
- Tổng wikilink: {total_links}
- Tổng entity: {len(entities)}
- Tổng relation: {len(relations)}

## Broken wikilink

{display(sorted(broken))}

## Entity trùng ID

{display(duplicates)}

## Trang có ID không tồn tại trong entities.csv

{display(unknown_pages)}

## Relation có endpoint không tồn tại

{display(relation_orphans)}

## Rủi ro chưa có kiểm soát

{display(no_controls)}

## Rủi ro chưa có sự kiện

{display(no_events)}

## Trang entity không có liên kết

{display(sorted(orphan_pages))}

## Phân loại kết quả

- Lỗi chương trình: {len(program_errors)}
- Khoảng trống dữ liệu (không tự động sửa): {len(no_controls) + len(no_events)}
- Trạng thái: {'ĐẠT' if not program_errors else 'CHƯA ĐẠT'}
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(f"Report: {OUTPUT}")
    print(f"Markdown={len(pages)}, wikilinks={total_links}, program_errors={len(program_errors)}")
    if program_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
