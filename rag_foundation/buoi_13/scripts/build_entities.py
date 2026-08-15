"""Normalize source CSVs to outputs/entities.csv and outputs/relations.csv."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
SOURCES = [
    ("risk_profiles_seed.csv", "RuiRo"),
    ("controls_seed.csv", "KiemSoat"),
    ("risk_events_seed.csv", "SuKienRuiRo"),
]
BASE_COLUMNS = ["id", "type", "name", "description", "source_file", "data_origin", "verification_status"]
RELATION_COLUMNS = ["source_id", "relationship_type", "target_id", "source", "evidence_quote", "confidence", "verification_status", "data_origin"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    entities: list[dict[str, str]] = []
    extra_columns: list[str] = []
    for filename, entity_type in SOURCES:
        for raw in read_rows(DATA / filename):
            for column in raw:
                if column not in BASE_COLUMNS and column not in extra_columns:
                    extra_columns.append(column)
            row = dict(raw)
            row.update(type=entity_type, source_file=filename)
            if entity_type == "SuKienRuiRo":
                row["name"] = raw.get("description", "")
            entities.append(row)

    ids = [row["id"] for row in entities]
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate entity IDs: {duplicates}")

    relations = read_rows(DATA / "relationships_seed.csv")
    entity_ids = set(ids)
    orphans = [r for r in relations if r["source_id"] not in entity_ids or r["target_id"] not in entity_ids]
    if orphans:
        raise ValueError(f"Orphan relationship references: {orphans}")

    write_rows(OUTPUTS / "entities.csv", BASE_COLUMNS + extra_columns, entities)
    write_rows(OUTPUTS / "relations.csv", RELATION_COLUMNS, relations)
    print(f"Wrote {len(entities)} entities and {len(relations)} relations")
    for key, count in sorted(Counter(row["type"] for row in entities).items()):
        print(f"  {key}: {count}")
    for key, count in sorted(Counter(row["relationship_type"] for row in relations).items()):
        print(f"  {key}: {count}")
    print("Orphan references: none")


if __name__ == "__main__":
    main()
