"""Vector seed retrieval and configurable multi-hop expansion in Neo4j."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from neo4j_connection import Neo4jConfig


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DOCUMENT_NUMBER = re.compile(
    r"\b\d{1,4}/(?:\d{4}/)?[A-ZĐÀ-Ỹ0-9]+(?:-[A-ZĐÀ-Ỹ0-9]+)+\b", re.IGNORECASE
)
DEFAULT_MODEL = "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5"
ALLOWED_RELATIONSHIPS = frozenset({
    "CAN_CU", "THAY_THE", "HOP_NHAT", "SUA_DOI_BO_SUNG", "VAN_BAN_BO_SUNG"
})


class RetrievalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievalConfig:
    vector_index: str = "chunk_embedding_index"
    text_property: str = "text"
    id_property: str = "id"
    model_name: str = DEFAULT_MODEL
    top_k: int = 5
    max_hops: int = 2
    relationships: tuple[str, ...] = (
        "CAN_CU", "THAY_THE", "HOP_NHAT", "SUA_DOI_BO_SUNG", "VAN_BAN_BO_SUNG"
    )

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        relationships = tuple(
            item.strip().upper()
            for item in os.getenv("GRAPH_RELATIONSHIPS", "CAN_CU,THAY_THE,HOP_NHAT,SUA_DOI_BO_SUNG,VAN_BAN_BO_SUNG").split(",")
            if item.strip()
        )
        config = cls(
            vector_index=os.getenv("NEO4J_VECTOR_INDEX", "chunk_embedding_index").strip(),
            text_property=os.getenv("NEO4J_TEXT_PROPERTY", "text").strip(),
            id_property=os.getenv("NEO4J_ID_PROPERTY", "id").strip(),
            model_name=os.getenv("MSMARCO_MODEL", DEFAULT_MODEL).strip(),
            top_k=int(os.getenv("TOP_K", "5")),
            max_hops=int(os.getenv("MAX_HOPS", "2")),
            relationships=relationships,
        )
        config.validate()
        return config

    def validate(self) -> None:
        for value in (self.vector_index, self.text_property, self.id_property):
            if not IDENTIFIER.fullmatch(value):
                raise RetrievalConfigError(f"Ten Neo4j khong hop le: {value!r}")
        if not 1 <= self.top_k <= 100:
            raise RetrievalConfigError("top_k phai trong khoang 1..100")
        if not 0 <= self.max_hops <= 5:
            raise RetrievalConfigError("max_hops phai trong khoang 0..5")
        invalid = set(self.relationships) - ALLOWED_RELATIONSHIPS
        if not self.relationships or invalid:
            raise RetrievalConfigError(f"Relationship khong duoc phep: {sorted(invalid)}")


class VietnameseMSMARCOEmbedder:
    """Lazy-loaded 384-dimensional Vietnamese MS MARCO sentence encoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Any = None

    def encode(self, question: str) -> list[float]:
        if not question or not question.strip():
            raise ValueError("Cau hoi khong duoc rong")
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("Chua cai sentence-transformers") from exc
            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode(question.strip(), normalize_embeddings=True)
        return [float(value) for value in vector]


def _driver_factory(config: Neo4jConfig):
    from neo4j import GraphDatabase
    return GraphDatabase.driver(config.uri, auth=(config.user, config.password))


def extract_document_numbers(question: str) -> list[str]:
    """Extract explicit Vietnamese legal document numbers, preserving order."""
    seen: set[str] = set()
    values: list[str] = []
    for match in DOCUMENT_NUMBER.finditer(question.upper()):
        value = match.group(0)
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def search_context(
    question: str,
    *,
    neo4j_config: Neo4jConfig | None = None,
    retrieval_config: RetrievalConfig | None = None,
    embedder: Any = None,
    driver_factory: Callable[[Neo4jConfig], Any] = _driver_factory,
) -> dict[str, Any]:
    """Return vector seeds plus unique nodes reached within 0..N relationship hops."""
    neo = neo4j_config or Neo4jConfig.from_env()
    cfg = retrieval_config or RetrievalConfig.from_env()
    cfg.validate()
    vector = (embedder or VietnameseMSMARCOEmbedder(cfg.model_name)).encode(question)
    explicit_documents = extract_document_numbers(question)
    relation_pattern = "|".join(f"`{name}`" for name in cfg.relationships)
    driver = driver_factory(neo)
    try:
        with driver.session(database=neo.database) as session:
            seed_rows = [dict(row) for row in session.run(
                "CALL db.index.vector.queryNodes($index, $top_k, $embedding) "
                "YIELD node, score OPTIONAL MATCH (node)-[:PART_OF]->(document:Document) "
                "RETURN elementId(node) AS element_id, "
                f"node.`{cfg.id_property}` AS id, node.`{cfg.text_property}` AS text, "
                "labels(node) AS labels, score, document.id AS document_id, "
                "document.so_ky_hieu AS document_number, document.title AS document_title "
                "ORDER BY score DESC",
                index=cfg.vector_index, top_k=cfg.top_k, embedding=vector,
            )]
            if explicit_documents:
                exact_rows = [dict(row) for row in session.run(
                    "MATCH (node:Chunk)-[:PART_OF]->(document:Document) "
                    "WHERE toUpper(document.so_ky_hieu) IN $document_numbers "
                    "AND node.embedding IS NOT NULL "
                    "WITH node, document, vector.similarity.cosine(node.embedding, $embedding) AS score "
                    "ORDER BY score DESC LIMIT $exact_k "
                    "RETURN elementId(node) AS element_id, node.id AS id, node.text AS text, "
                    "labels(node) AS labels, score, document.id AS document_id, "
                    "document.so_ky_hieu AS document_number, document.title AS document_title",
                    document_numbers=explicit_documents, embedding=vector,
                    exact_k=max(cfg.top_k * 2, cfg.top_k),
                )]
                merged = exact_rows + seed_rows
                seed_rows = []
                seen_seed_ids: set[str] = set()
                for row in merged:
                    if row["element_id"] not in seen_seed_ids:
                        seen_seed_ids.add(row["element_id"])
                        seed_rows.append(row)
                    if len(seed_rows) >= max(cfg.top_k * 2, cfg.top_k):
                        break
            seed_element_ids = [row["element_id"] for row in seed_rows]
            expanded_rows: list[dict[str, Any]] = []
            if cfg.max_hops and seed_element_ids:
                query = (
                    "MATCH (seed:Chunk) WHERE elementId(seed) IN $seed_ids "
                    "MATCH (seed)-[:PART_OF]->(seed_document:Document) "
                    f"MATCH path=(seed_document)-[:{relation_pattern}*1..{cfg.max_hops}]-(related_document:Document) "
                    "MATCH (related:Chunk)-[:PART_OF]->(related_document) "
                    "WHERE related.embedding IS NOT NULL "
                    "WITH related, related_document, min(length(path)) AS hop, "
                    "max(vector.similarity.cosine(related.embedding, $embedding)) AS related_score, "
                    "head(collect(elementId(seed))) AS seed_element_id, "
                    "head(collect([r IN relationships(path) | type(r)])) AS relationship_path "
                    "ORDER BY related_score DESC "
                    "RETURN seed_element_id, elementId(related) AS element_id, "
                    f"related.`{cfg.id_property}` AS id, related.`{cfg.text_property}` AS text, "
                    "labels(related) AS labels, hop, relationship_path, related_score, "
                    "related_document.id AS document_id, related_document.so_ky_hieu AS document_number, "
                    "related_document.title AS document_title "
                    "LIMIT $expanded_k"
                )
                expanded_rows = [dict(row) for row in session.run(
                    query, seed_ids=seed_element_ids, embedding=vector, expanded_k=cfg.top_k * 4
                )]
            context_element_ids = seed_element_ids + [row["element_id"] for row in expanded_rows]
            hierarchy_rows = [dict(row) for row in session.run(
                "MATCH (anchor:Chunk) WHERE elementId(anchor) IN $anchor_ids "
                "MATCH (parent:Chunk)-[:PARENT_OF]->(anchor) "
                "MATCH (parent)-[:PARENT_OF]->(sibling:Chunk) "
                "MATCH (sibling)-[:PART_OF]->(document:Document) "
                "RETURN DISTINCT elementId(sibling) AS element_id, sibling.id AS id, "
                "sibling.text AS text, labels(sibling) AS labels, parent.id AS parent_id, "
                "parent.title AS parent_title, document.id AS document_id, "
                "document.so_ky_hieu AS document_number, document.title AS document_title "
                "ORDER BY parent.id, sibling.id LIMIT $hierarchy_k",
                anchor_ids=context_element_ids, hierarchy_k=max(cfg.top_k * 12, 40),
            )] if context_element_ids else []
            document_ids = list({row.get("document_id") for row in seed_rows + expanded_rows
                                 if row.get("document_id")})
            graph_facts = [dict(row) for row in session.run(
                "MATCH (source:Document)-[r]->(target:Document) "
                "WHERE source.id IN $document_ids OR target.id IN $document_ids "
                "RETURN source.id AS source_id, source.so_ky_hieu AS source_number, "
                "source.title AS source_title, type(r) AS relationship, "
                "target.id AS target_id, target.so_ky_hieu AS target_number, "
                "target.title AS target_title",
                document_ids=document_ids,
            )] if document_ids else []
    finally:
        driver.close()

    seen = set(seed_element_ids)
    related = []
    for row in expanded_rows:
        if row["element_id"] not in seen:
            seen.add(row["element_id"])
            related.append(row)
    hierarchy = []
    for row in hierarchy_rows:
        if row["element_id"] not in seen:
            seen.add(row["element_id"])
            hierarchy.append(row)
    return {
        "question": question,
        "model": cfg.model_name,
        "top_k": cfg.top_k,
        "max_hops": cfg.max_hops,
        "relationships": list(cfg.relationships),
        "explicit_documents": explicit_documents,
        "seeds": seed_rows,
        "related": related,
        "hierarchy": hierarchy,
        "graph_facts": graph_facts,
        "context": seed_rows + related + hierarchy,
    }
