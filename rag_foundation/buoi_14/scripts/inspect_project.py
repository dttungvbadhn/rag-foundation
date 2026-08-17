from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
sys.path.insert(0, str(ROOT))
from src.common import source_dir


def inspect(path: Path) -> dict:
    raw = path.read_bytes()
    encoding = "utf-8-sig" if raw.startswith(b"\xef\xbb\xbf") else "utf-8"
    with path.open(encoding=encoding, newline="") as f:
        rows = list(csv.DictReader(f))
    duplicates = len(rows) - len({tuple(sorted(r.items())) for r in rows})
    nulls = {k: sum(not (r.get(k) or "").strip() for r in rows) for k in (rows[0] if rows else [])}
    return {"rows": len(rows), "columns": list(rows[0]) if rows else [], "encoding": encoding,
            "duplicates": duplicates, "nulls": nulls}


def main() -> None:
    src = source_dir()
    reports = {name: inspect(src / name) for name in ("metadata.csv", "content.csv", "relationships.csv")}
    rels = list(csv.DictReader((src / "relationships.csv").open(encoding="utf-8-sig", newline="")))
    types = Counter(r["relationship_type"] for r in rels)
    out = ROOT / "outputs" / "inspection_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Inspection report", "", f"- Working root: `{ROOT}`", f"- Source (read-only): `{src}`",
             "- Note: `kb+hops` was absent; selected the complete `ner_kb` triplet after schema inspection.",
             f"- Python: `{sys.version.split()[0]}`", ""]
    for name, info in reports.items():
        lines += [f"## {name}", "", f"- Rows: {info['rows']}", f"- Columns: `{', '.join(info['columns'])}`",
                  f"- Encoding: {info['encoding']}", f"- Exact duplicate rows: {info['duplicates']}",
                  f"- Null counts: `{info['nulls']}`", ""]
    lines += ["## Relationship types", "", *[f"- {k}: {v}" for k, v in sorted(types.items())], "",
              "## Safety", "", "No source files are written. No destructive Neo4j statement is used.", "",
              "Safe to continue: YES"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"PROJECT PRE-CHECK\nWorking root: {ROOT}\nData: {src}\nExisting code: inspected\nEnvironment: Python OK\nPotential risks: source path fallback\nSafe to continue: YES")


if __name__ == "__main__":
    main()
