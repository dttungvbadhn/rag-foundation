"""Offline tests for parent reranking and the four-mode answer pipeline."""

from dataclasses import replace
import unittest
from unittest.mock import Mock

from rag_foundation.Buoi_09 import advanced_rag as ar
from rag_foundation.Buoi_09 import hierarchical_rag as hr


def hcfg(**changes):
    return replace(hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH), **changes)


def acfg(**changes):
    return replace(ar.load_advanced_config(hr.ENV_EXAMPLE_PATH), **changes)


def parent_candidate(parent_id, rank, text=None, ambiguous=False):
    return {
        "parent_id": parent_id, "source": "law.pdf", "page_start": rank,
        "page_end": rank + 1,
        "structural_path": {"chapter": "I", "article": f"Điều {rank}",
                            "clause": None, "point": None},
        "text": text or f"parent text {parent_id}", "parent_rrf_score": 0.1 / rank,
        "parent_rank": rank, "anchor_child_id": f"child-{parent_id}",
        "scoring_child_ids": [f"child-{parent_id}"],
        "supporting_child_ids": [f"child-{parent_id}"],
        "support_query_ids": ["Q0"], "best_child_rank": rank,
        "ambiguous": ambiguous, "warnings": ["ambiguous"] if ambiguous else [],
    }


def fused_child(chunk_id="c1", rank=1):
    return {
        "chunk_id": chunk_id, "text": f"child text {chunk_id}", "source": "law.pdf",
        "page_start": 1, "page_end": 1, "bm25_rank": rank, "bm25_score": 2.0,
        "semantic_rank": rank, "semantic_distance": 0.1, "rrf_score": 0.03,
        "fused_rank": rank,
    }


def parent_retrieval_result(score=0.9, status="ready", ambiguous=False):
    item = parent_candidate("p1", 1, ambiguous=ambiguous)
    item.update({"parent_rerank_raw_score": 2.0, "parent_rerank_score": score,
                 "parent_rerank_rank": 1, "parent_rank_change": 0})
    return {
        "status": status, "mode": "multi_parent",
        "query_set": {"queries": [
            {"query_id": "Q0", "text": "original", "origin": "original"},
            {"query_id": "Q1", "text": "variant", "origin": "generated"},
        ]},
        "child_hits": [{"child_id": "child-p1"}],
        "parent_candidates": [parent_candidate("p1", 1)], "candidates": [item],
        "warnings": [],
        "trace": {"api_calls": {"generation_expansion": 1,
                                  "generation_answer": 0, "embedding": 2},
                  "identity": {"corpus_identity": "abc"}},
    }


class ParentRerankTests(unittest.TestCase):
    def test_pairs_use_original_question_and_parent_text_only(self):
        captured = []

        def scorer(pairs, config):
            captured.extend(pairs)
            return [0.0, 1.0]

        result = hr.rerank_parent_candidates(
            "Câu hỏi gốc", [parent_candidate("p1", 1, "A"),
                             parent_candidate("p2", 2, "B")],
            hcfg(), acfg(), scorer,
        )
        self.assertEqual([("Câu hỏi gốc", "A"), ("Câu hỏi gốc", "B")], captured)
        self.assertNotIn("variant", str(captured))
        self.assertEqual("reranked", result["status"])

    def test_sort_rank_change_final_k_and_sigmoid(self):
        result = hr.rerank_parent_candidates(
            "Q", [parent_candidate("p1", 1), parent_candidate("p2", 2),
                  parent_candidate("p3", 3)],
            hcfg(final_parent_top_k=2), acfg(), lambda pairs, config: [0.0, 2.0, 1.0],
        )
        self.assertEqual(["p2", "p3"], [item["parent_id"] for item in result["parents"]])
        self.assertEqual(1, result["parents"][0]["parent_rerank_rank"])
        self.assertEqual(1, result["parents"][0]["parent_rank_change"])
        self.assertAlmostEqual(1 / (1 + __import__("math").exp(-2)),
                               result["parents"][0]["parent_rerank_score"])

    def test_reranker_failure_has_explicit_status_without_fallback(self):
        def fail(*args):
            raise RuntimeError("model unavailable")

        result = hr.rerank_parent_candidates(
            "Q", [parent_candidate("p1", 1)], hcfg(), acfg(), fail
        )
        self.assertEqual("reranker_unavailable", result["status"])
        self.assertEqual([], result["parents"])


class AnswerPipelineTests(unittest.TestCase):
    def test_parent_gate_rejects_and_does_not_generate(self):
        generator = Mock(return_value="không được gọi")
        result = hr.answer_complete(
            "original", "multi_parent", hcfg(), acfg(rerank_min_score=0.8), [],
            generate=generator, retrieval_fn=lambda *args: parent_retrieval_result(0.7),
        )
        self.assertEqual("insufficient_evidence", result["status"])
        generator.assert_not_called()
        self.assertEqual(1, result["trace"]["api_calls"]["generation_expansion"])

    def test_parent_answer_citation_maps_real_parent_and_anchor(self):
        prompts = []

        def generate(prompt, config):
            prompts.append(prompt)
            return "Nội dung có căn cứ [P1]"

        result = hr.answer_complete(
            "original", "multi_parent", hcfg(), acfg(), [], generate=generate,
            retrieval_fn=lambda *args: parent_retrieval_result(0.9, ambiguous=True),
        )
        self.assertEqual("answered", result["status"])
        citation = result["citations"][0]
        self.assertEqual("p1", citation["parent_id"])
        self.assertEqual("child-p1", citation["anchor_child_id"])
        self.assertTrue(citation["ambiguous"])
        self.assertIn("CÂU HỎI GỐC:\noriginal", prompts[0])
        self.assertNotIn("variant", prompts[0])
        self.assertEqual(2, sum(result["trace"]["api_calls"][key]
                                for key in ("generation_expansion", "generation_answer")))

    def test_invalid_parent_label_prevents_answered_status(self):
        result = hr.answer_complete(
            "original", "multi_parent", hcfg(), acfg(), [],
            generate=lambda prompt, config: "Bịa label [P99]",
            retrieval_fn=lambda *args: parent_retrieval_result(0.9),
        )
        self.assertEqual("retrieval_only", result["status"])
        self.assertEqual([], result["citations"])
        self.assertTrue(any("P99" in warning for warning in result["warnings"]))

    def test_generated_query_never_enters_generation_prompt(self):
        seen = []
        result = hr.answer_complete(
            "original", "multi_parent", hcfg(), acfg(), [],
            generate=lambda prompt, config: seen.append(prompt) or "Đúng [P1]",
            retrieval_fn=lambda *args: parent_retrieval_result(0.9),
        )
        self.assertEqual("answered", result["status"])
        self.assertNotIn("variant", seen[0])

    def test_multi_query_failure_status_propagates_without_generation(self):
        generator = Mock()
        failed = parent_retrieval_result()
        failed.update({"status": "query_generation_unavailable", "candidates": [],
                       "warnings": ["expansion failed"]})
        result = hr.answer_complete(
            "original", "multi_parent", hcfg(), acfg(), [], generate=generator,
            retrieval_fn=lambda *args: failed,
        )
        self.assertEqual("query_generation_unavailable", result["status"])
        generator.assert_not_called()

    def test_reranker_failure_status_propagates(self):
        failed = parent_retrieval_result()
        failed.update({"status": "reranker_unavailable", "candidates": [],
                       "warnings": ["reranker failed"]})
        result = hr.answer_complete(
            "original", "single_parent", hcfg(), acfg(), [],
            retrieval_fn=lambda *args: failed,
        )
        self.assertEqual("reranker_unavailable", result["status"])

    def test_flat_mode_uses_baseline_evidence_gate_and_citations(self):
        item = fused_child()
        item.update({"rerank_raw_score": 2.0, "rerank_score": 0.9,
                     "rerank_rank": 1, "rank_change": 0})
        retrieved = {
            "status": "ready", "mode": "single_flat", "query_set": {"queries": []},
            "child_hits": [item], "parent_candidates": [], "candidates": [item],
            "warnings": [], "trace": {"api_calls": {"generation_expansion": 0,
                                                       "generation_answer": 0,
                                                       "embedding": 1}},
        }
        result = hr.answer_complete(
            "original", "single_flat", hcfg(), acfg(), [],
            generate=lambda prompt, config: "Căn cứ [E1]",
            retrieval_fn=lambda *args: retrieved,
        )
        self.assertEqual("answered", result["status"])
        self.assertEqual("c1", result["citations"][0]["chunk_id"])


class RoutingAndCompareTests(unittest.TestCase):
    def test_single_flat_routes_one_hybrid_and_one_reranker(self):
        hybrid = Mock(return_value={"candidates": [fused_child()], "trace": {}})
        scorer = Mock(return_value=[1.0])
        result = hr.retrieve_complete_mode(
            "Q", "single_flat", hcfg(), acfg(), [], hybrid_retriever=hybrid,
            score_pairs=scorer,
        )
        self.assertEqual("ready", result["status"])
        hybrid.assert_called_once()
        scorer.assert_called_once()
        self.assertEqual(0, result["trace"]["api_calls"]["generation_expansion"])
        self.assertIn("identity", result["trace"])

    def test_multi_flat_routes_expansion_hybrid_per_query_and_one_reranker(self):
        hr.clear_query_cache()
        generator = Mock(return_value={"queries": [
            {"text": "variant one", "focus": "paraphrase"},
        ]})
        hybrid = Mock(side_effect=lambda question, *args, **kwargs: {
            "candidates": [fused_child("c1", 1)], "trace": {}})
        scorer = Mock(return_value=[1.0])
        result = hr.retrieve_complete_mode(
            "original", "multi_flat", hcfg(), acfg(), [], generator, hybrid, scorer
        )
        self.assertEqual("ready", result["status"])
        self.assertEqual(2, hybrid.call_count)
        scorer.assert_called_once()
        self.assertEqual(1, result["trace"]["api_calls"]["generation_expansion"])

    def test_compare_runs_four_modes_and_never_generation(self):
        retrieval = Mock(side_effect=lambda question, mode, *args, **kwargs: {
            "status": "ready", "mode": mode, "candidates": [], "trace": {}})
        result = hr.compare_complete_modes(
            "Q", hcfg(), acfg(), [], retrieval_fn=retrieval
        )
        self.assertEqual(set(hr.ANSWER_MODES), set(result["modes"]))
        self.assertEqual(4, retrieval.call_count)
        self.assertEqual(0, result["generation_answer_calls"])


if __name__ == "__main__":
    unittest.main()
