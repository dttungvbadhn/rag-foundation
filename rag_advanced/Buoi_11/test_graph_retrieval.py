import unittest

from graph_retrieval import (RetrievalConfig, RetrievalConfigError, extract_document_numbers,
                             search_context)
from neo4j_connection import Neo4jConfig


class FakeEmbedder:
    def encode(self, question):
        self.question = question
        return [0.1, 0.2, 0.3]


class FakeResult(list):
    pass


class FakeSession:
    def __init__(self):
        self.calls = []
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def run(self, query, **params):
        self.calls.append((query, params))
        if "queryNodes" in query:
            return FakeResult([{"element_id": "4:a", "id": "A", "text": "seed", "labels": ["Chunk"], "score": .9}])
        return FakeResult([
            {"seed_element_id": "4:a", "element_id": "4:b", "id": "B", "text": "related", "labels": ["Chunk"], "hop": 1, "relationship_path": ["CAN_CU"]},
            {"seed_element_id": "4:a", "element_id": "4:b", "id": "B", "text": "related", "labels": ["Chunk"], "hop": 2, "relationship_path": ["CAN_CU", "HOP_NHAT"]},
        ])


class FakeDriver:
    def __init__(self): self.fake_session = FakeSession(); self.closed = False
    def session(self, database): self.database = database; return self.fake_session
    def close(self): self.closed = True


class RetrievalTests(unittest.TestCase):
    def test_vector_then_multihop_and_deduplicate(self):
        driver = FakeDriver()
        cfg = RetrievalConfig(top_k=3, max_hops=2)
        result = search_context(
            "quy dinh nao?", neo4j_config=Neo4jConfig("bolt://localhost:7687", "kb-hops", "neo4j", "x"),
            retrieval_config=cfg, embedder=FakeEmbedder(), driver_factory=lambda _: driver,
        )
        self.assertEqual(["A", "B"], [row["id"] for row in result["context"]])
        self.assertIn("*1..2", driver.fake_session.calls[1][0])
        self.assertEqual([0.1, 0.2, 0.3], driver.fake_session.calls[0][1]["embedding"])
        self.assertTrue(driver.closed)

    def test_zero_hops_only_runs_vector_query(self):
        driver = FakeDriver()
        result = search_context(
            "x", neo4j_config=Neo4jConfig("bolt://x", "kb-hops", "u", "p"),
            retrieval_config=RetrievalConfig(max_hops=0), embedder=FakeEmbedder(),
            driver_factory=lambda _: driver,
        )
        self.assertEqual(2, len(driver.fake_session.calls))  # vector + hierarchy sibling lookup
        self.assertEqual([], result["related"])

    def test_rejects_cypher_identifier_and_excessive_hops(self):
        with self.assertRaises(RetrievalConfigError):
            RetrievalConfig(vector_index="x`) MATCH (n) //").validate()
        with self.assertRaises(RetrievalConfigError):
            RetrievalConfig(max_hops=6).validate()

    def test_extracts_explicit_document_numbers(self):
        self.assertEqual(
            ["52/VBHN-NHNN", "41/2016/TT-NHNN"],
            extract_document_numbers("VBHN số 52/VBHN-NHNN và Thông tư 41/2016/TT-NHNN"),
        )


if __name__ == "__main__": unittest.main()
