"""Inspect the four source CSV files without changing them."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILES = {
    "risk_profiles_seed.csv": "id",
    "controls_seed.csv": "id",
    "risk_events_seed.csv": "id",
    "relationships_seed.csv": None,
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def main() -> None:
    loaded: dict[str, list[dict[str, str]]] = {}
    print("WIKI RISK GRAPH - DATA INSPECTION")
    for filename, primary_key in FILES.items():
        columns, rows = read_csv(DATA / filename)
        loaded[filename] = rows
        print(f"\n[{filename}]\nrows: {len(rows)}\ncolumns: {', '.join(columns)}")
        nulls = {column: sum(not (row.get(column) or "").strip() for row in rows) for column in columns}
        print("nulls: " + ", ".join(f"{key}={value}" for key, value in nulls.items()))
        if primary_key:
            counts = Counter(row[primary_key] for row in rows)
            duplicates = sorted(key for key, count in counts.items() if key and count > 1)
            print(f"primary_key: {primary_key}; duplicates: {duplicates or 'none'}")
        else:
            keys = [(r["source_id"], r["relationship_type"], r["target_id"]) for r in rows]
            duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
            print(f"composite_key: source_id+relationship_type+target_id; duplicates: {duplicates or 'none'}")

    risks = {r["id"] for r in loaded["risk_profiles_seed.csv"]}
    controls = {r["id"] for r in loaded["controls_seed.csv"]}
    events = {r["id"] for r in loaded["risk_events_seed.csv"]}
    entity_ids = risks | controls | events
    event_orphans = sorted(r["risk_id"] for r in loaded["risk_events_seed.csv"] if r["risk_id"] not in risks)
    relations = loaded["relationships_seed.csv"]
    relation_orphans = sorted(
        f"{r['source_id']} -{r['relationship_type']}-> {r['target_id']}"
        for r in relations
        if r["source_id"] not in entity_ids or r["target_id"] not in entity_ids
    )
    print("\n[REFERENCES]")
    print(f"risk_events.risk_id missing: {event_orphans or 'none'}")
    print(f"relationship endpoints missing: {relation_orphans or 'none'}")
    print("relationship_type: " + ", ".join(sorted({r["relationship_type"] for r in relations})))
    print("\nNodes: RuiRo, KiemSoat, SuKienRuiRo")
    print("Edges: KiemSoat -MITIGATES-> RuiRo; RuiRo -OBSERVED_AS-> SuKienRuiRo")
    print("Missing master data: owner_unit_id names and owner_role_id names; no units/roles/processes/documents/clauses tables.")


if __name__ == "__main__":
    main()
