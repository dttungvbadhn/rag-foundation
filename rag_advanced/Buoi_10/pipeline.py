"""Buổi 10: hierarchical chunking, embeddings and idempotent Neo4j import."""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HEADING_RE = re.compile(r"^(chương|mục|điều|khoản)\s+([\divxlcdm]+)\b", re.IGNORECASE)
ALLOWED_RELATIONSHIPS = {"CAN_CU", "THAY_THE", "HOP_NHAT", "VAN_BAN_BO_SUNG"}


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    level: str
    order: int
    parent_id: str | None


def clean_text(value: object) -> str:
    """Remove markup and normalize whitespace without rewriting legal text."""
    soup = BeautifulSoup("" if pd.isna(value) else str(value), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", unicodedata.normalize("NFC", line)).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _level(text: str) -> str:
    match = HEADING_RE.match(text)
    return match.group(1).lower() if match else "đoạn"


def chunk_document(document_id: object, html: object) -> list[Chunk]:
    """Create a conservative Chương→Mục→Điều→Khoản→Đoạn hierarchy."""
    doc_id = str(document_id).strip()
    lines = clean_text(html).splitlines()
    chunks: list[Chunk] = []
    parents: dict[str, str] = {}
    ranks = {"chương": 0, "mục": 1, "điều": 2, "khoản": 3, "đoạn": 4}
    for order, line in enumerate(lines):
        level = _level(line)
        rank = ranks[level]
        parent_id = None
        for candidate, candidate_rank in sorted(ranks.items(), key=lambda item: item[1], reverse=True):
            if candidate_rank < rank and candidate in parents:
                parent_id = parents[candidate]
                break
        chunk_id = f"{doc_id}:chunk:{order:05d}"
        chunks.append(Chunk(chunk_id, doc_id, line, level, order, parent_id))
        if level != "đoạn":
            parents[level] = chunk_id
            for child, child_rank in ranks.items():
                if child_rank > rank:
                    parents.pop(child, None)
    return chunks


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(DATA_DIR / "metadata.csv", dtype=str).fillna("")
    content = pd.read_csv(DATA_DIR / "content.csv", dtype=str).fillna("")
    relationships = pd.read_csv(DATA_DIR / "relationships.csv", dtype=str).fillna("")
    for name, frame in (("metadata", metadata), ("content", content)):
        if "id" not in frame or frame["id"].duplicated().any():
            raise ValueError(f"{name}.csv phải có id duy nhất")
    missing = set(metadata["id"]) ^ set(content["id"])
    if missing:
        raise ValueError(f"ID không khớp giữa metadata/content: {sorted(missing)}")
    return metadata, content, relationships


def build_chunks(content: pd.DataFrame) -> list[Chunk]:
    return [chunk for row in content.itertuples(index=False) for chunk in chunk_document(row.id, row.content_html)]


def embed_chunks(chunks: list[Chunk], model_name: str) -> list[list[float]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True, show_progress_bar=True)
    return [[float(value) for value in vector] for vector in vectors]


def import_graph(metadata: pd.DataFrame, chunks: list[Chunk], vectors: list[list[float]], relationships: pd.DataFrame) -> None:
    from neo4j import GraphDatabase

    load_dotenv(BASE_DIR / ".env")
    uri, user, password = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "kb-hops")
    if not all((uri, user, password)):
        raise RuntimeError("Thiếu NEO4J_URI, NEO4J_USER hoặc NEO4J_PASSWORD")
    documents = metadata.to_dict("records")
    chunk_rows = [{**c.__dict__, "embedding": vector} for c, vector in zip(chunks, vectors)]
    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE").consume()
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE").consume()
            session.run("UNWIND $rows AS row MERGE (d:Document {id: row.id}) SET d += row", rows=documents).consume()
            session.run(
                "UNWIND $rows AS row MERGE (c:Chunk {id: row.id}) "
                "SET c.text=row.text, c.level=row.level, c.order=row.order, c.embedding=row.embedding "
                "WITH c,row MATCH (d:Document {id:row.document_id}) MERGE (c)-[:PART_OF]->(d)", rows=chunk_rows
            ).consume()
            session.run(
                "UNWIND $rows AS row WITH row WHERE row.parent_id IS NOT NULL "
                "MATCH (p:Chunk {id:row.parent_id}),(c:Chunk {id:row.id}) MERGE (p)-[:PARENT_OF]->(c)", rows=chunk_rows
            ).consume()
            session.run(
                "UNWIND $rows AS row MATCH (a:Chunk {id:row.a}),(b:Chunk {id:row.b}) MERGE (a)-[:NEXT]->(b)",
                rows=[{"a": a.id, "b": b.id} for a, b in zip(chunks, chunks[1:]) if a.document_id == b.document_id]
            ).consume()
            for rel_type in ALLOWED_RELATIONSHIPS:
                rows = relationships[relationships["relationship_type"].eq(rel_type)].to_dict("records")
                if rows:
                    session.run(
                        f"UNWIND $rows AS row MATCH (a:Document {{id:row.doc_id}}),(b:Document {{id:row.other_doc_id}}) "
                        f"MERGE (a)-[r:{rel_type}]->(b) SET r.evidence=row.relationship", rows=rows
                    ).consume()
            session.run(
                "CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig:{`vector.dimensions`:384,`vector.similarity_function`:'cosine'}}"
            ).consume()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--import-neo4j", action="store_true")
    args = parser.parse_args()
    metadata, content, relationships = load_inputs()
    chunks = build_chunks(content)
    print(f"Documents: {len(metadata)}; chunks: {len(chunks)}")
    for chunk in chunks[:10]:
        print(chunk)
    if args.import_neo4j:
        load_dotenv(BASE_DIR / ".env")
        model = os.getenv("EMBEDDING_MODEL", "thuannc/vi-distilled-msmarco-MiniLM-L12-cos-v5")
        import_graph(metadata, chunks, embed_chunks(chunks, model), relationships)
        print("[PASS] Neo4j import")
    elif not args.dry_run:
        parser.error("Chọn --dry-run hoặc --import-neo4j")


if __name__ == "__main__":
    main()
