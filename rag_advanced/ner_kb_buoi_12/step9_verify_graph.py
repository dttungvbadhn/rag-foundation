from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATASET = "ner_kb_buoi_12"


def as_dicts(records: list[object]) -> list[dict[str, object]]:
    return [dict(record) for record in records]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_DIR / ".env")
    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    if not all((uri, user, password)):
        raise ValueError("Thiếu cấu hình Neo4j")

    documents = pd.read_csv(BASE_DIR / "cleaned_documents.csv", dtype={"id": "string"})
    entities = pd.read_csv(BASE_DIR / "entities.csv", dtype={"source_doc_id": "string"})
    relationships = pd.read_csv(
        BASE_DIR / "relationships.csv", dtype={"source": "string", "target": "string"}
    )
    expected_nodes = {"Document": len(documents)}
    expected_nodes.update(
        entities.drop_duplicates("entity_id")["entity_type"].value_counts().to_dict()
    )
    expected_relationships = relationships["relationship_type"].value_counts().to_dict()

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        query = lambda cypher: driver.execute_query(
            cypher,
            dataset=DATASET,
            database_=database,
            routing_="r",
        ).records
        node_counts = as_dicts(
            query(
                """
                MATCH (n)
                WHERE n.dataset = $dataset
                UNWIND labels(n) AS label
                RETURN label, count(*) AS total
                ORDER BY total DESC
                """
            )
        )
        relationship_counts = as_dicts(
            query(
                """
                MATCH (a)-[r]->(b)
                WHERE a.dataset = $dataset AND r.dataset = $dataset AND b.dataset = $dataset
                RETURN type(r) AS relationship_type, count(*) AS total
                ORDER BY total DESC
                """
            )
        )
        graph_sample_count = query(
            """
            MATCH (n)-[r]->(m)
            WHERE n.dataset = $dataset AND r.dataset = $dataset AND m.dataset = $dataset
            RETURN count(*) AS total
            LIMIT 100
            """
        )[0]["total"]
        signer_samples = as_dicts(
            query(
                """
                MATCH (d:Document)-[:KY_BOI]->(p:NguoiKy)
                WHERE d.dataset = $dataset
                RETURN d.so_ky_hieu AS document, p.canonical_name AS nguoi_ky
                LIMIT 5
                """
            )
        )
        applicable_samples = as_dicts(
            query(
                """
                MATCH (d:Document)-[:AP_DUNG_CHO]->(o:DoiTuongApDung)
                WHERE d.dataset = $dataset
                RETURN d.so_ky_hieu AS document, o.canonical_name AS doi_tuong
                LIMIT 5
                """
            )
        )
        document_relation_samples = as_dicts(
            query(
                """
                MATCH (a:Document)-[r:THAM_CHIEU|SUA_DOI_BO_SUNG|THAY_THE_BOI]->(b:Document)
                WHERE a.dataset = $dataset AND b.dataset = $dataset
                RETURN a.so_ky_hieu AS source, type(r) AS relationship_type,
                       b.so_ky_hieu AS target
                LIMIT 10
                """
            )
        )
        reference_chains = as_dicts(
            query(
                """
                MATCH path=(d1:Document)-[:THAM_CHIEU*1..3]->(d2:Document)
                WHERE d1.dataset = $dataset AND d2.dataset = $dataset
                RETURN [n IN nodes(path) | n.so_ky_hieu] AS documents,
                       length(path) AS hops
                LIMIT 20
                """
            )
        )
    finally:
        driver.close()

    actual_nodes = {row["label"]: row["total"] for row in node_counts}
    actual_relationships = {
        row["relationship_type"]: row["total"] for row in relationship_counts
    }
    nodes_match = actual_nodes == expected_nodes
    relationships_match = actual_relationships == expected_relationships
    required_queries_nonempty = all(
        (
            graph_sample_count > 0,
            bool(signer_samples),
            bool(applicable_samples),
            bool(document_relation_samples),
        )
    )

    print(f"database={database}")
    print(f"expected_nodes={json.dumps(expected_nodes, ensure_ascii=False, sort_keys=True)}")
    print(f"actual_nodes={json.dumps(actual_nodes, ensure_ascii=False, sort_keys=True)}")
    print(f"nodes_match_csv={nodes_match}")
    print(
        "expected_relationships="
        + json.dumps(expected_relationships, ensure_ascii=False, sort_keys=True)
    )
    print(
        "actual_relationships="
        + json.dumps(actual_relationships, ensure_ascii=False, sort_keys=True)
    )
    print(f"relationships_match_csv={relationships_match}")
    print(f"graph_sample_edges_available={graph_sample_count}")
    print(f"signer_samples={json.dumps(signer_samples, ensure_ascii=False)}")
    print(f"applicable_samples={json.dumps(applicable_samples, ensure_ascii=False)}")
    print(
        "document_relation_samples="
        + json.dumps(document_relation_samples, ensure_ascii=False)
    )
    print(f"reference_chains={json.dumps(reference_chains, ensure_ascii=False)}")
    passed = nodes_match and relationships_match and required_queries_nonempty
    print(f"STEP_9={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
