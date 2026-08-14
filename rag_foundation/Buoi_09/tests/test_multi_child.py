"""Offline tests for per-query retrieval and cross-query RRF."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from rag_foundation.Buoi_09 import hierarchical_rag as hr


def cfg(**changes):
    return replace(hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH), **changes)


def candidate(child_id, rank, text=None, source="doc.pdf"):
    return {
        "chunk_id": child_id,
        "text": text or f"Nội dung {child_id}",
        "source": source,
        "page_start": 1,
        "page_end": 2,
        "bm25_rank": rank,
        "bm25_score": 9.0 - rank,
        "semantic_rank": rank + 1,
        "semantic_distance": 0.1 * rank,
        "rrf_score": 0.02,
        "fused_rank": rank,
    }


def query(query_id, origin):
    return {"query_id": query_id, "text": query_id, "origin": origin,
            "focus": "original_intent" if origin == "original" else "paraphrase"}


def generator_payload():
    return {"queries": [
        {"text": "biến thể thứ nhất", "focus": "paraphrase"},
        {"text": "thuật ngữ pháp lý", "focus": "exact_legal_terms"},
    ]}


class CrossQueryRRFTests(unittest.TestCase):
    def setUp(self):
        hr.clear_query_cache()

    def test_formula_weights_union_support_and_missing_contribution(self):
        config = cfg(multi_query_rrf_k=10, original_weight=2.0, variant_weight=1.0)
        results = [
            {"query": query("Q0", "original"), "candidates": [candidate("a", 2), candidate("b", 1)]},
            {"query": query("Q1", "generated"), "candidates": [candidate("a", 1), candidate("c", 2)]},
        ]
        fused = hr.cross_query_rrf(results, config)
        by_id = {item["child_id"]: item for item in fused}
        self.assertAlmostEqual(2 / 12 + 1 / 11, by_id["a"]["multi_query_rrf_score"])
        self.assertAlmostEqual(2 / 11, by_id["b"]["multi_query_rrf_score"])
        self.assertEqual(3, len(fused))
        self.assertEqual(2, by_id["a"]["support_query_count"])
        self.assertEqual(["Q0", "Q1"], by_id["a"]["support_query_ids"])
        self.assertEqual({"Q0": 2, "Q1": 1}, by_id["a"]["per_query_ranks"])
        self.assertEqual(["Q1"], by_id["c"]["support_query_ids"])

    def test_metadata_mismatch_fails(self):
        with self.assertRaisesRegex(hr.HierarchyError, "Metadata mismatch"):
            hr.cross_query_rrf([
                {"query": query("Q0", "original"), "candidates": [candidate("a", 1)]},
                {"query": query("Q1", "generated"), "candidates": [candidate("a", 1, source="other.pdf")]},
            ], cfg())

    def test_deterministic_tie_break_uses_child_id(self):
        results = [{"query": query("Q0", "original"),
                    "candidates": [candidate("b", 1), candidate("a", 1)]}]
        first = hr.cross_query_rrf(results, cfg())
        second = hr.cross_query_rrf(results, cfg())
        self.assertEqual(["a", "b"], [item["child_id"] for item in first])
        self.assertEqual(first, second)


class MultiChildPipelineTests(unittest.TestCase):
    def setUp(self):
        hr.clear_query_cache()

    def test_each_query_calls_hybrid_once_and_trace_is_complete(self):
        fake_generator = Mock(return_value=generator_payload())
        seen = []

        def fake_hybrid(question, strategy, chunks, advanced_config, **options):
            seen.append(question)
            return {"candidates": [candidate("shared", 1), candidate(question, 2)],
                    "trace": {"inner": True}}

        result = hr.multi_query_child_retrieval(
            "câu hỏi gốc", cfg(), object(), [], fake_generator, fake_hybrid
        )
        self.assertEqual("ready", result["status"])
        self.assertEqual(3, len(seen))
        self.assertEqual(3, len(result["per_query_results"]))
        self.assertEqual(
            ["Q0", "Q1", "Q2"],
            [item["query"]["query_id"] for item in result["per_query_results"]],
        )
        self.assertTrue(all(item["candidates"] for item in result["per_query_results"]))
        self.assertEqual(3, result["trace"]["query_count_executed"])
        self.assertEqual(0, result["trace"]["query_count_failed"])
        self.assertEqual(3, result["trace"]["semantic_embedding_call_count"])
        self.assertEqual(1, result["trace"]["gemini_expansion_call_count"])
        self.assertEqual(4, result["trace"]["union_child_count"])
        self.assertEqual({"1": 3, "3": 1}, result["trace"]["overlap_distribution"])
        self.assertEqual({"Q0", "Q1", "Q2"}, set(result["trace"]["retrieval_latency_ms"]))
        self.assertGreaterEqual(result["trace"]["fusion_latency_ms"], 0)
        fake_generator.assert_called_once()

    def test_per_query_limit_and_inner_trace_are_preserved(self):
        config = cfg(per_query_candidates=1)
        retriever = Mock(return_value={
            "candidates": [candidate("a", 1), candidate("b", 2)],
            "trace": {"bm25_candidate_count": 2},
        })
        result = hr.multi_query_child_retrieval(
            "gốc", config, object(), [], Mock(return_value=generator_payload()), retriever
        )
        self.assertEqual(1, len(result["children"]))
        self.assertIn("bm25_rank", result["children"][0]["per_query_trace"]["Q0"])

    def test_q0_failure_stops_pipeline(self):
        retriever = Mock(side_effect=RuntimeError("Q0 hỏng"))
        result = hr.multi_query_child_retrieval(
            "gốc", cfg(), object(), [], Mock(return_value=generator_payload()), retriever
        )
        self.assertEqual("q0_retrieval_failed", result["status"])
        self.assertEqual(1, retriever.call_count)
        self.assertEqual(1, result["trace"]["query_count_executed"])
        self.assertEqual(1, result["trace"]["query_count_failed"])

    def test_some_generated_failure_is_partial(self):
        calls = 0

        def retriever(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Q1 hỏng")
            return {"candidates": [candidate(f"c{calls}", 1)], "trace": {}}

        result = hr.multi_query_child_retrieval(
            "gốc", cfg(), object(), [], Mock(return_value=generator_payload()), retriever
        )
        self.assertEqual("partial", result["status"])
        self.assertEqual(3, result["trace"]["query_count_executed"])
        self.assertEqual(1, result["trace"]["query_count_failed"])
        self.assertNotIn("Q1", result["trace"]["result_count_by_query"])

    def test_all_generated_failures_are_multi_query_partial(self):
        calls = 0

        def retriever(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("variant hỏng")
            return {"candidates": [candidate("q0", 1)], "trace": {}}

        result = hr.multi_query_child_retrieval(
            "gốc", cfg(), object(), [], Mock(return_value=generator_payload()), retriever
        )
        self.assertEqual("multi_query_partial", result["status"])
        self.assertEqual(["Q0"], result["children"][0]["support_query_ids"])
        self.assertEqual(2, result["trace"]["query_count_failed"])

    def test_no_reranker_or_generation_boundary_exists(self):
        retriever = Mock(return_value={"candidates": [candidate("a", 1)], "trace": {}})
        result = hr.multi_query_child_retrieval(
            "gốc", cfg(), object(), [], Mock(return_value=generator_payload()), retriever
        )
        self.assertNotIn("answer", result)
        self.assertNotIn("rerank_score", result["children"][0])
        self.assertEqual(3, retriever.call_count)


if __name__ == "__main__":
    unittest.main()
