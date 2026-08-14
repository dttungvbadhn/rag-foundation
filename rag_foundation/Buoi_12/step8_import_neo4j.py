from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DOCUMENTS_PATH = BASE_DIR / "cleaned_documents.csv"
ENTITIES_PATH = BASE_DIR / "entities.csv"
RELATIONSHIPS_PATH = BASE_DIR / "relationships.csv"
ERRORS_PATH = BASE_DIR / "import_errors.csv"
DATASET = "ner_kb_buoi_12"

ENTITY_LABELS = {
    "CoQuan": "CoQuan",
    "NguoiKy": "NguoiKy",
    "DoiTuongApDung": "DoiTuongApDung",
    "LinhVuc": "LinhVuc",
}
ENTITY_RELATION_TARGETS = {
    "BAN_HANH_BOI": "CoQuan",
    "KY_BOI": "NguoiKy",
    "AP_DUNG_CHO": "DoiTuongApDung",
    "THUOC_LINH_VUC": "LinhVuc",
}
DOCUMENT_RELATIONS = {"THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI"}
ALLOWED_RELATIONS = DOCUMENT_RELATIONS | set(ENTITY_RELATION_TARGETS)


def clean_value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def records(frame: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    selected = frame if columns is None else frame[columns]
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in selected.to_dict("records")
    ]


def create_constraints(driver: Any, database: str) -> None:
    constraints = {
        "Document": "ner_kb_document_id",
        "CoQuan": "ner_kb_co_quan_id",
        "NguoiKy": "ner_kb_nguoi_ky_id",
        "DoiTuongApDung": "ner_kb_doi_tuong_id",
        "LinhVuc": "ner_kb_linh_vuc_id",
    }
    for label, name in constraints.items():
        key = "id" if label == "Document" else "entity_id"
        driver.execute_query(
            f"CREATE CONSTRAINT {name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE",
            database_=database,
        )


def import_once(
    driver: Any,
    database: str,
    documents: pd.DataFrame,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    document_columns = [
        column for column in documents.columns if column not in {"content_html"}
    ]
    try:
        driver.execute_query(
            """
            UNWIND $rows AS row
            MERGE (d:Document {id: row.id})
            SET d += row, d.dataset = $dataset
            """,
            rows=records(documents, document_columns),
            dataset=DATASET,
            database_=database,
        )
    except Exception as exc:
        errors.append({"stage": "documents", "error": f"{type(exc).__name__}: {exc}"})

    canonical = entities.sort_values("confidence", ascending=False).drop_duplicates("entity_id")
    for entity_type, label in ENTITY_LABELS.items():
        subset = canonical[canonical["entity_type"].eq(entity_type)]
        try:
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MERGE (n:{label} {{entity_id: row.entity_id}})
                SET n.canonical_name = row.canonical_name,
                    n.entity_type = row.entity_type,
                    n.dataset = $dataset
                """,
                rows=records(subset, ["entity_id", "canonical_name", "entity_type"]),
                dataset=DATASET,
                database_=database,
            )
        except Exception as exc:
            errors.append({"stage": f"entities:{entity_type}", "error": f"{type(exc).__name__}: {exc}"})

    relation_columns = [
        "source", "target", "relationship_type", "method", "confidence", "evidence"
    ]
    for relation_type in sorted(ALLOWED_RELATIONS):
        subset = relationships[relationships["relationship_type"].eq(relation_type)]
        if subset.empty:
            continue
        if relation_type in DOCUMENT_RELATIONS:
            match_target = "MATCH (t:Document {id: row.target})"
        else:
            target_label = ENTITY_RELATION_TARGETS[relation_type]
            match_target = f"MATCH (t:{target_label} {{entity_id: row.target}})"
        try:
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (s:Document {{id: row.source}})
                {match_target}
                MERGE (s)-[r:{relation_type}]->(t)
                SET r.method = row.method,
                    r.confidence = row.confidence,
                    r.evidence = row.evidence,
                    r.dataset = $dataset
                """,
                rows=records(subset, relation_columns),
                dataset=DATASET,
                database_=database,
            )
        except Exception as exc:
            errors.append({"stage": f"relationships:{relation_type}", "error": f"{type(exc).__name__}: {exc}"})
    return errors


def graph_counts(driver: Any, database: str) -> dict[str, dict[str, int]]:
    node_records = driver.execute_query(
        """
        MATCH (n {dataset: $dataset})
        UNWIND labels(n) AS label
        RETURN label, count(*) AS total
        ORDER BY label
        """,
        dataset=DATASET,
        database_=database,
        routing_="r",
    ).records
    relationship_records = driver.execute_query(
        """
        MATCH (a {dataset: $dataset})-[r {dataset: $dataset}]->(b {dataset: $dataset})
        RETURN type(r) AS relationship_type, count(*) AS total
        ORDER BY relationship_type
        """,
        dataset=DATASET,
        database_=database,
        routing_="r",
    ).records
    return {
        "nodes": {record["label"]: record["total"] for record in node_records},
        "relationships": {
            record["relationship_type"]: record["total"]
            for record in relationship_records
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(PROJECT_DIR / ".env")
    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip() or "neo4j"
    if not all((uri, user, password, database)):
        raise ValueError("Thiếu cấu hình Neo4j")

    documents = pd.read_csv(DOCUMENTS_PATH, dtype={"id": "string"})
    entities = pd.read_csv(ENTITIES_PATH, dtype={"source_doc_id": "string"})
    relationships = pd.read_csv(
        RELATIONSHIPS_PATH, dtype={"source": "string", "target": "string"}
    )
    driver = GraphDatabase.driver(uri, auth=(user, password))
    all_errors: list[dict[str, str]] = []
    try:
        driver.verify_connectivity()
        create_constraints(driver, database)
        first_errors = import_once(driver, database, documents, entities, relationships)
        all_errors.extend({"run": "first", **error} for error in first_errors)
        first_counts = graph_counts(driver, database)
        second_errors = import_once(driver, database, documents, entities, relationships)
        all_errors.extend({"run": "second", **error} for error in second_errors)
        second_counts = graph_counts(driver, database)
    finally:
        driver.close()

    pd.DataFrame(all_errors, columns=["run", "stage", "error"]).to_csv(
        ERRORS_PATH, index=False, encoding="utf-8-sig"
    )
    idempotent = first_counts == second_counts
    print(f"database={database}")
    print(f"first_counts={json.dumps(first_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"second_counts={json.dumps(second_counts, ensure_ascii=False, sort_keys=True)}")
    print(f"import_errors={len(all_errors)}")
    print(f"idempotent={idempotent}")
    passed = not all_errors and idempotent
    print(f"STEP_8={'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
