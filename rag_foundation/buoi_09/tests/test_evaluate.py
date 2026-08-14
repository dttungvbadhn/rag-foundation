"""Offline deterministic evaluator tests; no service/model/storage boundary."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from rag_foundation.buoi_09 import advanced_rag as ar
from rag_foundation.buoi_09 import evaluate
from rag_foundation.buoi_09 import hierarchical_rag as hr


class MetricTests(unittest.TestCase):
    def test_recall_mrr_ndcg_binary(self):
        ranked = ["x", "a", "b"]
        relevant = {"a", "b"}
        self.assertEqual(1.0, evaluate.recall_at_k(ranked, relevant, 3))
        self.assertEqual(0.5, evaluate.reciprocal_rank_at_k(ranked, relevant, 3))
        expected_dcg = 1 / __import__("math").log2(3) + 1 / __import__("math").log2(4)
        ideal = 1 + 1 / __import__("math").log2(3)
        self.assertAlmostEqual(expected_dcg / ideal, evaluate.ndcg_at_k(ranked, relevant, 3))
        self.assertIsNone(evaluate.recall_at_k(ranked, set(), 3))


class EvaluationDataTests(unittest.TestCase):
    def test_snapshot_input_path_resolves_real_buoi05_data(self):
        self.assertEqual(hr.INPUT_DIR, ar.semantic_baseline.DEFAULT_INPUT_DIR)
        self.assertTrue(ar.semantic_baseline.DEFAULT_INPUT_DIR.is_dir())

    def test_labels_resolve_and_stale_id_fails(self):
        children = [{"child_id": "c1"}]
        parents = [{"parent_id": "p1"}]
        record = [{
            "question_id": "Q1", "question": "Question", "question_type": "exact",
            "relevant_child_ids": ["c1"], "relevant_parent_ids": ["p1"],
            "needs_human_review": True, "notes": "provisional",
        }]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "questions.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(record, evaluate.load_questions(path, children, parents))
            record[0]["relevant_parent_ids"] = ["stale"]
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(evaluate.EvaluationError, "stale"):
                evaluate.load_questions(path, children, parents)


class EvaluatorTests(unittest.TestCase):
    def test_four_modes_metrics_calls_and_no_answer_generation(self):
        children = [{"child_id": "c1", "parent_id": "p1"},
                    {"child_id": "c2", "parent_id": "p2"}]
        parents = [{"parent_id": "p1", "source": "law.pdf"},
                   {"parent_id": "p2", "source": "law.pdf"}]
        questions = [{
            "question_id": "Q1", "question": "Question", "question_type": "exact",
            "relevant_child_ids": ["c1"], "relevant_parent_ids": ["p1"],
            "needs_human_review": True, "notes": "provisional",
        }]
        calls = []

        def fake_retrieval(question, mode, *args, **kwargs):
            calls.append(mode)
            trace = {"total_latency_ms": 10.0,
                     "api_calls": {"generation_expansion": int(mode.startswith("multi")),
                                   "generation_answer": 0, "embedding": 2},
                     "identity": {"corpus_identity": "corpus"}}
            if mode.endswith("parent"):
                candidates = [{"parent_id": "p1", "supporting_child_ids": ["c1"],
                               "source": "law.pdf", "text": "parent context"}]
            else:
                candidates = [{"chunk_id": "c1", "source": "law.pdf", "text": "child"}]
            return {"status": "ready", "mode": mode, "candidates": candidates,
                    "child_hits": [{"child_id": "c1", "text": "child"}],
                    "query_set": {"queries": [{"query_id": "Q0"}]},
                    "warnings": [], "trace": trace}

        report = evaluate.evaluate_questions(
            questions, 5, hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH),
            ar.load_advanced_config(hr.ENV_EXAMPLE_PATH), [], children, parents,
            retrieval_fn=fake_retrieval,
        )
        self.assertEqual(set(hr.ANSWER_MODES), set(calls))
        self.assertEqual(0, report["answer_generation_calls"])
        self.assertTrue(report["needs_human_review"])
        for mode in hr.ANSWER_MODES:
            metrics = report["per_question"][0]["modes"][mode]["metrics"]
            self.assertEqual(1.0, metrics["child_recall_at_k"])
            self.assertEqual(1.0, metrics["parent_recall_at_k"])
            self.assertEqual(1.0, metrics["mrr_at_k"])
            self.assertEqual(1.0, metrics["ndcg_at_k"])

    def test_atomic_report_and_latest_are_equal(self):
        report = {"timestamp": "2026-01-01T00:00:00+00:00", "value": 1}
        with tempfile.TemporaryDirectory() as temp:
            report_path, latest_path = evaluate.write_report_atomic(report, Path(temp))
            self.assertTrue(report_path.is_file())
            self.assertTrue(latest_path.is_file())
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")),
                             json.loads(latest_path.read_text(encoding="utf-8")))
            self.assertFalse(any(path.suffix == ".tmp" for path in Path(temp).iterdir()))


if __name__ == "__main__":
    unittest.main()
