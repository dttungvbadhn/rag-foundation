"""Read-only verification for the Buổi 10 Neo4j graph."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


def main() -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    values = [os.getenv(name) for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")]
    if not all(values):
        raise RuntimeError("Thiếu cấu hình Neo4j trong .env")
    with GraphDatabase.driver(values[0], auth=(values[1], values[2])) as driver:
        driver.verify_connectivity()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "kb-hops")) as session:
            node_counts = session.run("MATCH (n) RETURN labels(n) AS labels,count(*) AS total ORDER BY total DESC").data()
            rel_counts = session.run("MATCH ()-[r]->() RETURN type(r) AS type,count(*) AS total ORDER BY total DESC").data()
    print("Nodes:", node_counts)
    print("Relationships:", rel_counts)
    print("[PASS] Neo4j verification")


if __name__ == "__main__":
    main()
