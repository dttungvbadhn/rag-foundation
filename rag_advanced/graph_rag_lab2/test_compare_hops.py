import unittest

from compare_hops import render_markdown, run_comparison


QUESTIONS = [{"question_id": "Q01", "question": "Câu hỏi?"}]


class ComparisonTests(unittest.TestCase):
    def test_same_question_runs_at_three_hops(self):
        seen = []
        def retrieve(question, retrieval_config):
            seen.append(retrieval_config.max_hops)
            return {"seeds": [{"id": "S", "text": "seed"}],
                    "related": ([{"id": "R", "text": "related", "hop": retrieval_config.max_hops,
                                  "relationship_path": ["CAN_CU"]}]
                                if retrieval_config.max_hops else [])}
        rows = run_comparison(QUESTIONS, generate=False, retriever=retrieve)
        self.assertEqual([0, 1, 2], seen)
        self.assertEqual([0, 1, 1], [row["related_count"] for row in rows])
        self.assertTrue(all(row["generation_calls"] == 0 for row in rows))

    def test_report_does_not_claim_success_for_failures(self):
        rows = [{"question_id": "Q01", "hops": hop, "status": "failed",
                 "error_type": "ServiceUnavailable", "error": "Neo4j offline",
                 "generation_calls": 0, "latency_ms": 1.0} for hop in (0, 1, 2)]
        report = render_markdown(QUESTIONS, rows, 5)
        self.assertIn("NOT RUN / INCOMPLETE", report)
        self.assertIn("Chưa thể chứng minh", report)


if __name__ == "__main__": unittest.main()
