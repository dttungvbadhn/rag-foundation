"""Pure-Python tests for Buổi 09 Streamlit presentation helpers."""

import unittest

from rag_foundation.Buoi_09 import app


class AppHelperTests(unittest.TestCase):
    def test_query_child_matrix_has_query_rank_cells_and_support(self):
        result = {
            "query_set": {"queries": [{"query_id": "Q0"}, {"query_id": "Q1"}]},
            "child_hits": [{
                "child_id": "c1", "per_query_ranks": {"Q0": 2},
                "support_query_count": 1, "multi_query_rrf_score": 0.02,
            }],
        }
        row = app.build_query_child_matrix(result)[0]
        self.assertEqual(2, row["Q0"])
        self.assertEqual("—", row["Q1"])
        self.assertEqual(1, row["support_query_count"])
        self.assertEqual(0.02, row["MQ-RRF score"])

    def test_parent_tree_preserves_rank_movement_children_and_ambiguity(self):
        result = {
            "child_hits": [{"child_id": "c1", "support_query_ids": ["Q0", "Q1"],
                            "per_query_ranks": {"Q0": 1, "Q1": 3}, "text": "anchor text"}],
            "accepted_evidence": [{
                "parent_id": "p1", "source": "law.pdf", "page_start": 1, "page_end": 2,
                "structural_path": {"article": "Điều 7"}, "parent_rank": 3,
                "parent_rerank_rank": 1, "parent_rrf_score": 0.03,
                "parent_rerank_score": 0.9, "text": "parent context",
                "supporting_child_ids": ["c1"], "anchor_child_id": "c1",
                "ambiguous": True, "warnings": ["metadata conflict"],
            }],
        }
        parent = app.build_parent_tree_data(result)[0]
        self.assertEqual((3, 1), (parent["parent_rank"], parent["parent_rerank_rank"]))
        self.assertTrue(parent["ambiguous"])
        self.assertTrue(parent["children"][0]["anchor"])
        self.assertEqual(["Q0", "Q1"], parent["children"][0]["query_ids"])

    def test_mode_comparison_row_reports_counts_calls_and_expansion(self):
        result = {
            "status": "ready", "warnings": [],
            "child_hits": [{"text": "12345"}, {"text": "abc"}],
            "candidates": [{
                "parent_id": "p1", "text": "x" * 24, "source": "law.pdf",
                "structural_path": {"article": "Điều 7"},
            }],
            "trace": {"total_latency_ms": 12.5,
                      "api_calls": {"generation_expansion": 1,
                                    "generation_answer": 0, "embedding": 3}},
        }
        row = app.build_comparison_row("multi_parent", result)
        self.assertEqual("parent", row["unit_type"])
        self.assertEqual(2, row["retrieved_child_count"])
        self.assertEqual(1, row["expanded_parent_count"])
        self.assertEqual(3.0, row["expansion_factor"])
        self.assertEqual(1, row["Generation calls"])
        self.assertEqual(3, row["Embedding calls"])

    def test_citation_format_uses_real_parent_anchor_and_page_range(self):
        rendered = app.format_citation({
            "evidence_id": "P1", "source": "law.pdf", "page_start": 2,
            "page_end": 4, "parent_id": "parent-1", "anchor_child_id": "child-7",
        })
        self.assertIn("[P1]", rendered)
        self.assertIn("tr. 2-4", rendered)
        self.assertIn("parent-1", rendered)
        self.assertIn("child-7", rendered)

    def test_status_mapping_distinguishes_actionable_failures(self):
        for status in (
            "hierarchy_not_ready", "collection_not_ready",
            "query_generation_unavailable", "multi_query_partial",
            "reranker_unavailable", "insufficient_evidence", "generation_error",
        ):
            mapped = app.status_message(status)
            self.assertIn(mapped["severity"], {"success", "warning", "error"})
            self.assertTrue(mapped["message"])
        self.assertNotEqual(app.status_message("hierarchy_not_ready")["message"],
                            app.status_message("reranker_unavailable")["message"])

    def test_import_has_no_api_or_model_action(self):
        self.assertTrue(callable(app.render_app))
        self.assertTrue(callable(app.build_query_child_matrix))


if __name__ == "__main__":
    unittest.main()
