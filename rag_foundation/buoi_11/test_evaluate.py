import json
import tempfile
import unittest
from pathlib import Path

from evaluate import evaluate, load_questions, write_report


class EvaluationTests(unittest.TestCase):
    def test_five_required_questions_load(self):
        questions = load_questions()
        self.assertEqual(["Q01", "Q02", "Q03", "Q04", "Q05"], [q["question_id"] for q in questions])

    def test_metrics_and_retrieval_only_budget(self):
        calls = []
        def fake_retriever(question, retrieval_config):
            calls.append((question, retrieval_config.max_hops))
            return {"seeds": [{"id": "S", "text": "seed"}],
                    "related": [{"id": "R", "text": "related", "hop": 2,
                                 "relationship_path": ["CAN_CU", "THAY_THE", "HOP_NHAT"]}]}
        report = evaluate(load_questions(), top_k=3, max_hops=2, retriever=fake_retriever)
        self.assertEqual(5, report["summary"]["successful"])
        self.assertEqual(0, report["summary"]["failed"])
        self.assertEqual(1, report["summary"]["mean_related_count"])
        self.assertTrue(all(row["generation_call_count"] == 0 for row in report["questions"]))
        self.assertEqual(5, len(calls))

    def test_failure_is_recorded_without_aborting_set(self):
        def failing(*_args, **_kwargs): raise RuntimeError("offline")
        report = evaluate(load_questions(), retriever=failing)
        self.assertEqual(5, report["summary"]["failed"])
        self.assertTrue(all(row["error_type"] == "RuntimeError" for row in report["questions"]))

    def test_atomic_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            write_report({"ok": True}, output)
            self.assertEqual({"ok": True}, json.loads(output.read_text(encoding="utf-8")))
            self.assertFalse(output.with_suffix(".json.tmp").exists())


if __name__ == "__main__": unittest.main()
