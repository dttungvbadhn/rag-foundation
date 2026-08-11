"""Metric và evaluator tests tính tay, offline."""

import math
import unittest

from rag_foundation.buoi_08 import advanced_rag
from rag_foundation.buoi_08 import evaluate


def config():
    return advanced_rag.AdvancedConfig(
        api_key="", embedding_model="embed", embedding_dim=128,
        generation_model="gen", max_distance=.45, bm25_candidates=3,
        semantic_candidates=3, rrf_k=60, rrf_bm25_weight=1,
        rrf_semantic_weight=1, rerank_candidates=3, final_top_k=2,
        reranker_model="reranker", reranker_max_length=512,
        rerank_batch_size=2, rerank_min_score=.5, rerank_device="cpu",
    )


class MetricTests(unittest.TestCase):
    def test_metrics_by_hand(self):
        ranking = ["x", "a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(1.0, evaluate.recall_at_k(ranking, relevant, 3))
        self.assertEqual(0.5, evaluate.mrr_at_k(ranking, relevant, 3))
        expected = ((1 / math.log2(3)) + (1 / math.log2(4))) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(expected, evaluate.ndcg_at_k(ranking, relevant, 3))

    def test_empty_relevance_is_zero(self):
        self.assertEqual(0, evaluate.recall_at_k(["a"], set(), 1))
        self.assertEqual(0, evaluate.mrr_at_k(["a"], set(), 1))
        self.assertEqual(0, evaluate.ndcg_at_k(["a"], set(), 1))

    def test_evaluator_no_generation_review_warning_and_query_failure(self):
        questions = [
            {"query_id": "Q1", "question": "ok", "relevant_chunk_ids": ["a"],
             "scope": "in_scope", "needs_human_review": True},
            {"query_id": "Q2", "question": "fail", "relevant_chunk_ids": ["b"],
             "scope": "in_scope", "needs_human_review": True},
        ]

        def retrieval(question, mode, strategy, chunks, cfg):
            if question == "fail":
                raise RuntimeError("expected")
            return {"status": "retrieved", "candidates": [{"chunk_id": "a"}],
                    "trace": advanced_rag._empty_trace()}

        report = evaluate.evaluate_retrieval(
            questions, ["bm25", "semantic"], "hierarchical", 1, [], config(), retrieval
        )
        self.assertTrue(report["needs_human_review"])
        self.assertTrue(report["warnings"])
        self.assertEqual(4, len(report["details"]))
        self.assertEqual(2, sum(item["status"] == "failed" for item in report["details"]))
        self.assertEqual(1.0, report["metrics"]["bm25"]["recall@1"])


if __name__ == "__main__":
    unittest.main()
