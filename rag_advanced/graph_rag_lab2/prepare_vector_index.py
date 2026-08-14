"""Idempotently embed Neo4j Chunk nodes and create the 384d cosine vector index."""

from __future__ import annotations

import argparse

from neo4j import GraphDatabase

from graph_retrieval import RetrievalConfig, VietnameseMSMARCOEmbedder
from neo4j_connection import Neo4jConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    neo, cfg = Neo4jConfig.from_env(), RetrievalConfig.from_env()
    driver = GraphDatabase.driver(neo.uri, auth=(neo.user, neo.password))
    embedder = VietnameseMSMARCOEmbedder(cfg.model_name)
    try:
        with driver.session(database=neo.database) as session:
            total = session.run(
                "MATCH (c:Chunk) WHERE c.text IS NOT NULL AND trim(c.text) <> '' RETURN count(c) AS n"
            ).single()["n"]
            completed = 0
            while True:
                rows = [dict(row) for row in session.run(
                    "MATCH (c:Chunk) WHERE c.text IS NOT NULL AND trim(c.text) <> '' "
                    "AND c.embedding IS NULL RETURN elementId(c) AS element_id, c.text AS text "
                    "LIMIT $limit", limit=args.batch_size
                )]
                if not rows:
                    break
                vectors = embedder._model.encode(
                    [row["text"] for row in rows], normalize_embeddings=True, batch_size=args.batch_size
                ) if embedder._model is not None else None
                if vectors is None:
                    embedder.encode(rows[0]["text"])
                    vectors = embedder._model.encode(
                        [row["text"] for row in rows], normalize_embeddings=True, batch_size=args.batch_size
                    )
                payload = [{"element_id": row["element_id"], "embedding": [float(x) for x in vector]}
                           for row, vector in zip(rows, vectors)]
                session.run(
                    "UNWIND $rows AS row MATCH (c:Chunk) WHERE elementId(c)=row.element_id "
                    "SET c.embedding=row.embedding", rows=payload
                ).consume()
                completed += len(rows)
                print(f"Embedded {completed}/{total}", flush=True)
            session.run(
                f"CREATE VECTOR INDEX `{cfg.vector_index}` IF NOT EXISTS FOR (c:Chunk) ON (c.embedding) "
                "OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}"
            ).consume()
            session.run("CALL db.awaitIndexes(300)").consume()
            count = session.run("MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS n").single()["n"]
            print(f"Vector index ready: {cfg.vector_index}; embedded chunks: {count}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__": raise SystemExit(main())
