"""Advanced answer pipeline tests: all external boundaries are mocked."""

import unittest
from unittest.mock import Mock

from rag_foundation.buoi_08 import advanced_rag


def config():
    return advanced_rag.AdvancedConfig(
        api_key="test", embedding_model="embed", embedding_dim=128,
        generation_model="generate", max_distance=0.45, bm25_candidates=20,
        semantic_candidates=20, rrf_k=60, rrf_bm25_weight=1,
        rrf_semantic_weight=1, rerank_candidates=20, final_top_k=5,
        reranker_model="reranker", reranker_max_length=512,
        rerank_batch_size=4, rerank_min_score=0.5, rerank_device="cpu",
    )


def candidate(chunk_id="a", distance=0.2, rerank_score=0.8, text="accepted text"):
    return {
        "chunk_id": chunk_id, "text": text, "source": "real.pdf",
        "page_start": 2, "page_end": 3, "bm25_rank": 1, "bm25_score": 4.2,
        "semantic_rank": 1, "semantic_distance": distance, "rrf_score": 0.03,
        "fused_rank": 1, "rerank_raw_score": 1.4,
        "rerank_score": rerank_score, "rerank_rank": 1, "rank_change": 0,
    }


def trace():
    value = advanced_rag._empty_trace()
    value.update({"bm25_candidates": 2, "semantic_candidates": 2,
                  "overlap": 1, "union": 3, "reranked": 2})
    return value


def retrieval_result(items, status="retrieved"):
    return {"status": status, "candidates": items, "warnings": [], "trace": trace()}


class AnswerPipelineTests(unittest.TestCase):
    def assert_schema(self, result):
        self.assertEqual(
            {"status", "mode", "question", "answer", "evidence", "citations", "warnings", "trace"},
            set(result),
        )
        self.assertEqual(
            {"bm25_candidates", "semantic_candidates", "overlap", "union", "reranked",
             "accepted", "generation_called", "latency_ms"}, set(result["trace"]),
        )
        self.assertEqual(
            {"bm25", "semantic", "fusion", "rerank", "generation", "total"},
            set(result["trace"]["latency_ms"]),
        )

    def test_mode_validation(self):
        with self.assertRaisesRegex(ValueError, "mode không hợp lệ"):
            advanced_rag.answer_advanced(
                "q", "invalid", "hierarchical", [], config(),
                retrieval=Mock(), generate=Mock(),
            )

    def test_mode_gating_and_rejected_evidence_not_in_prompt(self):
        for mode, good, bad in (
            ("semantic", candidate("good", 0.2, 0.1), candidate("bad", 0.8, 0.9, "rejected text")),
            ("bm25", candidate("good", 0.2, 0.1), candidate("bad", None, 0.9, "rejected text")),
            ("hybrid", candidate("good", 0.2, 0.1), candidate("bad", 0.8, 0.9, "rejected text")),
            ("hybrid_rerank", candidate("good", 0.9, 0.8), candidate("bad", 0.1, 0.2, "rejected text")),
        ):
            generator = Mock(return_value="Trả lời [E1]")
            retrieval = Mock(return_value=retrieval_result([good, bad]))
            result = advanced_rag.answer_advanced(
                "question", mode, "hierarchical", [], config(),
                retrieval=retrieval, generate=generator,
            )
            self.assertEqual("answered", result["status"])
            prompt = generator.call_args.args[0]
            self.assertIn("accepted text", prompt)
            self.assertNotIn(bad["text"], prompt)
            self.assertEqual([True, False], [item["accepted"] for item in result["evidence"]])
            self.assert_schema(result)

    def test_no_accepted_evidence_skips_generation_and_schema_complete(self):
        generator = Mock()
        result = advanced_rag.answer_advanced(
            "q", "semantic", "hierarchical", [], config(),
            retrieval=Mock(return_value=retrieval_result([candidate(distance=0.9)])),
            generate=generator,
        )
        self.assertEqual("insufficient_evidence", result["status"])
        generator.assert_not_called()
        self.assertFalse(result["trace"]["generation_called"])
        self.assert_schema(result)

    def test_citation_uses_real_metadata_and_fake_label_warns(self):
        result = advanced_rag.answer_advanced(
            "q", "hybrid_rerank", "hierarchical", [], config(),
            retrieval=Mock(return_value=retrieval_result([candidate()])),
            generate=Mock(return_value="Nội dung [E1], sai [E99]"),
        )
        self.assertEqual("real.pdf", result["citations"][0]["source"])
        self.assertEqual(2, result["citations"][0]["page_start"])
        self.assertNotIn("[E99]", result["answer"])
        self.assertTrue(result["warnings"])

    def test_generation_failure_or_empty_is_retrieval_only_once(self):
        for generator in (Mock(side_effect=OSError("secret")), Mock(return_value="  ")):
            result = advanced_rag.answer_advanced(
                "q", "semantic", "hierarchical", [], config(),
                retrieval=Mock(return_value=retrieval_result([candidate()])),
                generate=generator,
            )
            self.assertEqual("retrieval_only", result["status"])
            generator.assert_called_once()
            self.assertEqual([], result["citations"])
            self.assert_schema(result)

    def test_reranker_unavailable_has_separate_status_and_no_generation(self):
        generator = Mock()
        unavailable = retrieval_result([], "reranker_unavailable")
        unavailable["warnings"] = ["Reranker không khả dụng"]
        result = advanced_rag.answer_advanced(
            "q", "hybrid_rerank", "hierarchical", [], config(),
            retrieval=Mock(return_value=unavailable), generate=generator,
        )
        self.assertEqual("reranker_unavailable", result["status"])
        generator.assert_not_called()
        self.assert_schema(result)

    def test_compare_never_calls_generation_and_reports_modes(self):
        retrieval = Mock(side_effect=lambda question, mode, strategy, chunks, cfg: retrieval_result([candidate(mode)]))
        result = advanced_rag.compare_modes("q", "hierarchical", [], config(), retrieval=retrieval)
        self.assertEqual(4, retrieval.call_count)
        self.assertEqual(set(advanced_rag.ANSWER_MODES), {row["chunk_id"] for row in result["rows"]})
        self.assertEqual(set(advanced_rag.ANSWER_MODES), set(result["latency_ms"]))

    def test_compare_uses_mode_specific_final_rank(self):
        item = candidate("shared")
        item.update({"bm25_rank": 7, "semantic_rank": 2, "fused_rank": 4,
                     "rerank_rank": 1, "rank_change": 3})
        retrieval = Mock(return_value=retrieval_result([item]))
        result = advanced_rag.compare_modes("q", "hierarchical", [], config(), retrieval=retrieval)
        self.assertEqual(
            {"bm25": 7, "semantic": 2, "hybrid": 4, "hybrid_rerank": 1},
            result["rows"][0]["ranks"],
        )

    def test_evidence_has_all_nullable_stage_fields(self):
        sparse = candidate()
        sparse.update({"bm25_rank": None, "bm25_score": None, "rrf_score": None,
                       "fused_rank": None, "rerank_raw_score": None,
                       "rerank_score": None, "rerank_rank": None, "rank_change": None})
        result = advanced_rag.answer_advanced(
            "q", "semantic", "hierarchical", [], config(),
            retrieval=Mock(return_value=retrieval_result([sparse])),
            generate=Mock(return_value="ok"),
        )
        evidence = result["evidence"][0]
        for field in ("bm25_rank", "bm25_score", "semantic_rank", "semantic_distance",
                      "rrf_score", "fused_rank", "rerank_raw_score", "rerank_score",
                      "rerank_rank", "rank_change", "accepted"):
            self.assertIn(field, evidence)


if __name__ == "__main__":
    unittest.main()
