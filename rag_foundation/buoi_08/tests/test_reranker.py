"""Kiểm thử cross-encoder reranker hoàn toàn mock và offline."""

import math
import sys
import unittest
from unittest.mock import Mock, patch

from rag_foundation.buoi_08 import advanced_rag


def config(rerank_candidates=3, final_top_k=2):
    return advanced_rag.AdvancedConfig(
        api_key="test", embedding_model="embed", embedding_dim=128,
        generation_model="generate", max_distance=0.45, bm25_candidates=3,
        semantic_candidates=3, rrf_k=60, rrf_bm25_weight=1.0,
        rrf_semantic_weight=1.0, rerank_candidates=rerank_candidates,
        final_top_k=final_top_k, reranker_model="BAAI/bge-reranker-v2-m3",
        reranker_max_length=512, rerank_batch_size=2, rerank_min_score=0.5,
        rerank_device="cpu",
    )


def candidate(chunk_id, fused_rank):
    return {
        "chunk_id": chunk_id, "text": f"Nội dung {chunk_id}",
        "source": "sample.pdf", "page_start": 1, "page_end": 1,
        "bm25_rank": fused_rank, "bm25_score": 1.0,
        "semantic_rank": fused_rank, "semantic_distance": 0.1,
        "rrf_score": 0.03, "fused_rank": fused_rank,
        "matched_by": ["bm25", "semantic"],
    }


class RerankerTests(unittest.TestCase):
    def test_import_and_non_rerank_paths_do_not_load_ml_modules(self):
        self.assertNotIn("transformers.models.auto.tokenization_auto", sys.modules)
        with patch.object(advanced_rag, "_load_cross_encoder") as loader:
            advanced_rag.rerank_fused_candidates(
                "câu hỏi", [candidate("a", 1)], config(),
                score_pairs=lambda pairs, cfg: [1.0],
            )
        loader.assert_not_called()

    def test_one_pair_per_candidate_and_candidate_limit(self):
        scorer = Mock(return_value=[0.1, 0.2])
        result = advanced_rag.rerank_fused_candidates(
            "câu hỏi", [candidate("a", 1), candidate("b", 2), candidate("c", 3)],
            config(rerank_candidates=2), score_pairs=scorer,
        )
        pairs, passed_config = scorer.call_args.args
        self.assertEqual(2, len(pairs))
        self.assertEqual(["Nội dung a", "Nội dung b"], [pair[1] for pair in pairs])
        self.assertEqual(2, result["reranked_count"])
        self.assertIs(passed_config, scorer.call_args.args[1])

    def test_sigmoid_sort_tie_break_rank_change_and_final_top_k(self):
        result = advanced_rag.rerank_fused_candidates(
            "q", [candidate("b", 1), candidate("a", 2), candidate("c", 3)],
            config(final_top_k=2), score_pairs=lambda pairs, cfg: [0.0, 0.0, 2.0],
        )
        self.assertEqual("reranked", result["status"])
        self.assertEqual(["c", "b"], [item["chunk_id"] for item in result["candidates"]])
        top = result["candidates"][0]
        self.assertAlmostEqual(1 / (1 + math.exp(-2)), top["rerank_score"])
        self.assertEqual(1, top["rerank_rank"])
        self.assertEqual(2, top["rank_change"])
        self.assertEqual("BAAI/bge-reranker-v2-m3", top["reranker_model"])

    def test_batch_scorer_must_preserve_count(self):
        result = advanced_rag.rerank_fused_candidates(
            "q", [candidate("a", 1), candidate("b", 2)], config(),
            score_pairs=lambda pairs, cfg: [1.0],
        )
        self.assertEqual("reranker_unavailable", result["status"])
        self.assertEqual([], result["candidates"])

    def test_runtime_batching_preserves_all_logits_offline(self):
        class DeviceValue:
            def to(self, device):
                return self

        class FakeLogits:
            def __init__(self, values):
                self.values = values

            def reshape(self, *_args):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_args):
                return False

        batch_sizes = []

        def tokenizer(batch, **kwargs):
            batch_sizes.append(len(batch))
            self.assertEqual(512, kwargs["max_length"])
            return {"input_ids": DeviceValue()}

        model = Mock()
        model.side_effect = lambda **kwargs: type(
            "Output", (), {"logits": FakeLogits([float(len(batch_sizes))] * batch_sizes[-1])}
        )()
        fake_torch = type("FakeTorch", (), {"no_grad": staticmethod(lambda: NoGrad())})
        pairs = [("q", f"doc-{index}") for index in range(5)]
        with patch.dict(sys.modules, {"torch": fake_torch}), \
             patch.object(advanced_rag, "_resolve_reranker_device", return_value="cpu"), \
             patch.object(advanced_rag, "_load_cross_encoder", return_value=(tokenizer, model)):
            logits = advanced_rag._runtime_cross_encoder_logits(pairs, config())
        self.assertEqual([2, 2, 1], batch_sizes)
        self.assertEqual(5, len(logits))
        self.assertEqual(3, model.call_count)

    def test_model_failure_has_no_silent_rrf_fallback(self):
        def fail(pairs, cfg):
            raise OSError("offline")

        result = advanced_rag.rerank_fused_candidates(
            "q", [candidate("a", 1)], config(), score_pairs=fail,
        )
        self.assertEqual("reranker_unavailable", result["status"])
        self.assertEqual([], result["candidates"])
        self.assertIn("offline", result["warnings"][0])

    def test_hybrid_rerank_injection_never_loads_or_downloads_model(self):
        hybrid = Mock(return_value={
            "candidates": [candidate("a", 1)], "trace": {"union_count": 1}
        })
        with patch.object(advanced_rag, "_load_cross_encoder") as loader:
            result = advanced_rag.hybrid_rerank(
                "q", "hierarchical", [], config(),
                score_pairs=lambda pairs, cfg: [1.0], hybrid_retriever=hybrid,
            )
        loader.assert_not_called()
        hybrid.assert_called_once()
        self.assertEqual("reranked", result["status"])
        self.assertEqual({"union_count": 1}, result["trace"])


if __name__ == "__main__":
    unittest.main()
