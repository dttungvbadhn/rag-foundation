"""Kiểm thử offline Reciprocal Rank Fusion và hybrid trace."""

import unittest
from unittest.mock import Mock, patch

from rag_foundation.buoi_08 import advanced_rag
from rag_foundation.buoi_08 import rag


def bm25(chunk_id, rank, score=4.2, text=None, source="sample.pdf", start=1, end=1):
    return {
        "chunk_id": chunk_id,
        "text": text or f"text-{chunk_id}",
        "source": source,
        "page_start": start,
        "page_end": end,
        "bm25_rank": rank,
        "bm25_score": score,
    }


def semantic(chunk_id, rank, distance=0.2, text=None, source="sample.pdf", start=1, end=1):
    return {
        "chunk_id": chunk_id,
        "text": text or f"text-{chunk_id}",
        "source": source,
        "page_start": start,
        "page_end": end,
        "semantic_rank": rank,
        "semantic_distance": distance,
    }


def config(bm25_weight=1.0, semantic_weight=1.0):
    return advanced_rag.AdvancedConfig(
        api_key="test-key", embedding_model="model", embedding_dim=128,
        generation_model="generation", max_distance=0.45, bm25_candidates=20,
        semantic_candidates=20, rrf_k=60, rrf_bm25_weight=bm25_weight,
        rrf_semantic_weight=semantic_weight, rerank_candidates=20, final_top_k=5,
        reranker_model="reranker", reranker_max_length=512, rerank_batch_size=4,
        rerank_min_score=0.5, rerank_device="cpu",
    )


class RRFTests(unittest.TestCase):
    def test_formula_overlap_and_union_without_duplicates(self):
        results = advanced_rag.reciprocal_rank_fusion(
            [bm25("a", 1), bm25("b", 2)],
            [semantic("a", 3), semantic("c", 1)],
            60, 1.0, 2.0,
        )
        by_id = {item["chunk_id"]: item for item in results}
        self.assertEqual(3, len(results))
        self.assertAlmostEqual(1 / 61 + 2 / 63, by_id["a"]["rrf_score"])
        self.assertEqual(["bm25", "semantic"], by_id["a"]["matched_by"])
        self.assertEqual(["bm25"], by_id["b"]["matched_by"])
        self.assertEqual(["semantic"], by_id["c"]["matched_by"])

    def test_zero_weight_removes_that_branch_contribution(self):
        result = advanced_rag.reciprocal_rank_fusion(
            [bm25("a", 1)], [semantic("a", 1)], 60, 0.0, 2.0
        )[0]
        self.assertAlmostEqual(2 / 61, result["rrf_score"])

    def test_raw_scores_do_not_change_rrf(self):
        first = advanced_rag.reciprocal_rank_fusion(
            [bm25("a", 1, score=999)], [semantic("a", 2, distance=0.01)], 60, 1, 1
        )[0]["rrf_score"]
        second = advanced_rag.reciprocal_rank_fusion(
            [bm25("a", 1, score=-99)], [semantic("a", 2, distance=99)], 60, 1, 1
        )[0]["rrf_score"]
        self.assertEqual(first, second)

    def test_tie_break_is_deterministic(self):
        results = advanced_rag.reciprocal_rank_fusion(
            [bm25("b", 1), bm25("a", 1)], [], 60, 1, 0
        )
        self.assertEqual(["a", "b"], [item["chunk_id"] for item in results])
        self.assertEqual([1, 2], [item["fused_rank"] for item in results])

    def test_metadata_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "Metadata mismatch"):
            advanced_rag.reciprocal_rank_fusion(
                [bm25("a", 1, source="one.pdf")],
                [semantic("a", 1, source="two.pdf")], 60, 1, 1
            )

    def test_hybrid_trace_counts_and_each_retriever_called_once(self):
        lexical = [bm25("a", 1), bm25("b", 2)]
        dense = [semantic("a", 1), semantic("c", 2)]
        bm25_retriever = Mock(return_value=lexical)
        semantic_retriever = Mock(return_value=dense)
        result = advanced_rag.hybrid_retrieve(
            "question", "hierarchical", [], config(),
            bm25_retriever=bm25_retriever,
            semantic_retriever=semantic_retriever,
        )
        bm25_retriever.assert_called_once()
        semantic_retriever.assert_called_once()
        trace = result["trace"]
        self.assertEqual(2, trace["bm25_candidate_count"])
        self.assertEqual(2, trace["semantic_candidate_count"])
        self.assertEqual(3, trace["union_count"])
        self.assertEqual(1, trace["overlap_count"])
        self.assertEqual(3, trace["fused_count"])
        self.assertEqual({"tokenize_bm25", "semantic", "fusion"}, set(trace["latency_ms"]))

    def test_hybrid_does_not_load_reranker_or_generation(self):
        with patch.object(rag.genai, "Client") as gemini:
            result = advanced_rag.hybrid_retrieve(
                "question", "hierarchical", [], config(),
                bm25_retriever=Mock(return_value=[bm25("a", 1)]),
                semantic_retriever=Mock(return_value=[semantic("a", 1)]),
            )
        gemini.assert_not_called()
        self.assertNotIn("reranker_score", result["candidates"][0])
        self.assertNotIn("answer", result)


if __name__ == "__main__":
    unittest.main()
