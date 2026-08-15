"""Load normalized Wiki Risk Graph CSVs into Neo4j using parameterized Cypher."""

from __future__ import annotations

import csv
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    from neo4j import GraphDatabase
except ImportError as exc:
    raise SystemExit("Thiếu thư viện. Chạy: pip install -r requirements.txt") from exc

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"RuiRo", "KiemSoat", "SuKienRuiRo"}
RELATIONSHIPS = {
    "MITIGATES": ("KiemSoat", "RuiRo"),
    "OBSERVED_AS": ("RuiRo", "SuKienRuiRo"),
}


def rows(name: str) -> list[dict[str, str]]:
    with (ROOT / "outputs" / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    load_dotenv(ROOT / ".env")
    uri, user, password = (os.getenv(key) for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"))
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    if not all((uri, user, password)):
        raise SystemExit("Hãy cấu hình NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD và NEO4J_DATABASE trong .env")
    entities = rows("entities.csv")
    relations = rows("relations.csv")
    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
            with driver.session(database=database) as session:
                for entity in entities:
                    label = entity.pop("type")
                    if label not in LABELS:
                        raise ValueError(f"Unsupported label: {label}")
                    session.run(f"MERGE (n:{label} {{id: $id}}) SET n += $properties", id=entity["id"], properties=entity).consume()
                for relation in relations:
                    rel_type = relation["relationship_type"]
                    if rel_type not in RELATIONSHIPS:
                        raise ValueError(f"Unsupported relationship: {rel_type}")
                    source_label, target_label = RELATIONSHIPS[rel_type]
                    properties = {k: v for k, v in relation.items() if k not in {"source_id", "target_id", "relationship_type"}}
                    query = (f"MATCH (s:{source_label} {{id: $source_id}}), (t:{target_label} {{id: $target_id}}) "
                             f"MERGE (s)-[r:{rel_type}]->(t) SET r += $properties")
                    session.run(query, source_id=relation["source_id"], target_id=relation["target_id"], properties=properties).consume()
    except Exception as exc:
        raise SystemExit(f"Không thể nạp Neo4j: {exc}\nWiki và các output CSV không bị thay đổi.") from exc
    print(f"Loaded {len(entities)} nodes and {len(relations)} relationships into database {database}")


if __name__ == "__main__":
    main()
