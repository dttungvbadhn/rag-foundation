from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from neo4j import GraphDatabase


DATASET = "ner_kb_buoi_12"
DOCUMENT_RELATIONS = {"THAM_CHIEU", "SUA_DOI_BO_SUNG", "THAY_THE_BOI"}
RELATIONSHIP_TYPES = [
    "THAM_CHIEU",
    "SUA_DOI_BO_SUNG",
    "THAY_THE_BOI",
    "BAN_HANH_BOI",
    "KY_BOI",
    "AP_DUNG_CHO",
    "THUOC_LINH_VUC",
]


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


@dataclass(frozen=True)
class AppData:
    documents: pd.DataFrame
    entities: pd.DataFrame
    relationships: pd.DataFrame
    validation: pd.DataFrame


def load_app_data(base_dir: Path) -> AppData:
    documents = pd.read_csv(base_dir / "cleaned_documents.csv", dtype={"id": "string"})
    entities = pd.read_csv(base_dir / "entities.csv", dtype={"source_doc_id": "string"})
    relationships = pd.read_csv(
        base_dir / "relationships.csv", dtype={"source": "string", "target": "string"}
    )
    validation = pd.read_csv(
        base_dir / "validation_report.csv", dtype={"source": "string", "target": "string"}
    )
    return AppData(documents, entities, relationships, validation)


def search_documents(
    documents: pd.DataFrame,
    query: str = "",
    document_types: Iterable[str] = (),
    fields: Iterable[str] = (),
) -> pd.DataFrame:
    result = documents.copy()
    selected_types = list(document_types)
    selected_fields = list(fields)
    if selected_types:
        result = result[result["loai_van_ban"].isin(selected_types)]
    if selected_fields:
        result = result[result["linh_vuc"].fillna("Chưa phân loại").isin(selected_fields)]
    query = clean(query).casefold()
    if query:
        searchable = (
            result[["so_ky_hieu", "title", "content_clean"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.casefold()
        )
        result = result[searchable.str.contains(re.escape(query), regex=True)]
    return result.sort_values(["ngay_ban_hanh", "so_ky_hieu"], ascending=[False, True])


def document_bundle(data: AppData, document_id: str) -> dict[str, pd.DataFrame | pd.Series]:
    matches = data.documents[data.documents["id"].astype(str).eq(str(document_id))]
    if matches.empty:
        raise KeyError(f"Không tìm thấy document id={document_id}")
    document = matches.iloc[0]
    entities = data.entities[data.entities["source_doc_id"].astype(str).eq(str(document_id))]
    relations = data.relationships[
        data.relationships["source"].astype(str).eq(str(document_id))
        | data.relationships["target"].astype(str).eq(str(document_id))
    ].copy()

    document_names = data.documents.set_index("id")["so_ky_hieu"].astype(str).to_dict()
    entity_names = (
        data.entities.drop_duplicates("entity_id")
        .set_index("entity_id")["canonical_name"]
        .astype(str)
        .to_dict()
    )
    names = {**document_names, **entity_names}
    relations["source_name"] = relations["source"].map(names).fillna(relations["source"])
    relations["target_name"] = relations["target"].map(names).fillna(relations["target"])
    return {"document": document, "entities": entities, "relationships": relations}


def csv_statistics(data: AppData) -> dict[str, Any]:
    unique_entities = data.entities.drop_duplicates("entity_id")
    return {
        "documents": len(data.documents),
        "entities": len(unique_entities),
        "relationships": len(data.relationships),
        "entity_types": unique_entities["entity_type"].value_counts().to_dict(),
        "relationship_types": data.relationships["relationship_type"].value_counts().to_dict(),
        "validation": data.validation["status"].value_counts().to_dict(),
    }


class Neo4jRepository:
    def __init__(self, uri: str, user: str, password: str, database: str) -> None:
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def verify(self) -> bool:
        self.driver.verify_connectivity()
        record = self.driver.execute_query(
            "RETURN 1 AS ok", database_=self.database, routing_="r"
        ).records[0]
        return record["ok"] == 1

    def statistics(self) -> dict[str, dict[str, int]]:
        nodes = self.driver.execute_query(
            """
            MATCH (n) WHERE n.dataset = $dataset
            UNWIND labels(n) AS label
            RETURN label, count(*) AS total ORDER BY total DESC
            """,
            dataset=DATASET,
            database_=self.database,
            routing_="r",
        ).records
        relations = self.driver.execute_query(
            """
            MATCH (a)-[r]->(b)
            WHERE a.dataset = $dataset AND r.dataset = $dataset AND b.dataset = $dataset
            RETURN type(r) AS relationship_type, count(*) AS total ORDER BY total DESC
            """,
            dataset=DATASET,
            database_=self.database,
            routing_="r",
        ).records
        return {
            "nodes": {row["label"]: row["total"] for row in nodes},
            "relationships": {row["relationship_type"]: row["total"] for row in relations},
        }

    def graph(
        self,
        relationship_types: Iterable[str],
        limit: int = 100,
        document_id: str | None = None,
        hops: int = 1,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        selected = [item for item in relationship_types if item in RELATIONSHIP_TYPES]
        if not selected:
            return [], []
        limit = max(1, min(int(limit), 300))
        hops = max(1, min(int(hops), 3))
        if document_id:
            cypher = f"""
            MATCH p=(center:Document {{id: $document_id}})-[*1..{hops}]-(n)
            WHERE center.dataset = $dataset AND n.dataset = $dataset
            UNWIND relationships(p) AS r
            WITH DISTINCT startNode(r) AS a, r, endNode(r) AS b
            WHERE type(r) IN $types AND r.dataset = $dataset
            RETURN a, r, b LIMIT $limit
            """
            parameters = {"document_id": str(document_id)}
        else:
            cypher = """
            MATCH (a)-[r]->(b)
            WHERE a.dataset = $dataset AND r.dataset = $dataset AND b.dataset = $dataset
              AND type(r) IN $types
            RETURN a, r, b LIMIT $limit
            """
            parameters = {}
        result = self.driver.execute_query(
            cypher,
            dataset=DATASET,
            types=selected,
            limit=limit,
            database_=self.database,
            routing_="r",
            **parameters,
        ).records

        nodes: dict[str, dict[str, str]] = {}
        edges = []
        for row in result:
            for key in ("a", "b"):
                node = row[key]
                labels = list(node.labels)
                node_type = next((label for label in labels if label != "Entity"), labels[0])
                node_id = str(node.get("id") or node.get("entity_id"))
                display = clean(
                    node.get("so_ky_hieu")
                    or node.get("canonical_name")
                    or node.get("title")
                    or node_id
                )
                nodes[node_id] = {"id": node_id, "label": display, "type": node_type}
            relationship = row["r"]
            source_id = str(row["a"].get("id") or row["a"].get("entity_id"))
            target_id = str(row["b"].get("id") or row["b"].get("entity_id"))
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "type": relationship.type,
                    "evidence": clean(relationship.get("evidence")),
                }
            )
        return list(nodes.values()), edges


NODE_COLORS = {
    "Document": "#2563EB",
    "CoQuan": "#16A34A",
    "NguoiKy": "#9333EA",
    "DoiTuongApDung": "#EA580C",
    "LinhVuc": "#0891B2",
}


def dot_graph(nodes: list[dict[str, str]], edges: list[dict[str, str]]) -> str:
    def quote(value: object) -> str:
        return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'

    lines = [
        "digraph KnowledgeGraph {",
        "rankdir=LR;",
        'graph [bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.7"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontcolor="white"];',
        'edge [fontname="Arial", fontsize=9, color="#64748B"];',
    ]
    for node in nodes:
        color = NODE_COLORS.get(node["type"], "#475569")
        label = node["label"] if len(node["label"]) <= 55 else node["label"][:52] + "..."
        tooltip = html.unescape(node["label"])
        lines.append(
            f"{quote(node['id'])} [label={quote(label)}, fillcolor={quote(color)}, "
            f"tooltip={quote(tooltip)}];"
        )
    for edge in edges:
        lines.append(
            f"{quote(edge['source'])} -> {quote(edge['target'])} "
            f"[label={quote(edge['type'])}, tooltip={quote(edge['evidence'][:300])}];"
        )
    lines.append("}")
    return "\n".join(lines)
