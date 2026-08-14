"""Offline tests for child-to-parent mapping, Parent-RRF and context budget."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from rag_foundation.Buoi_09 import hierarchical_rag as hr


def cfg(**changes):
    return replace(hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH), **changes)


def child(child_id, parent_id, text=None, ambiguous=False):
    return {
        "child_id": child_id, "parent_id": parent_id, "source": "doc.pdf",
        "page_start": 1, "page_end": 1, "text": text or f"child {child_id}",
        "structural_path": {"chapter": "I", "article": parent_id,
                            "clause": None, "point": None},
        "resolution_method": "metadata", "ambiguous": ambiguous,
        "warnings": ["ambiguous_child"] if ambiguous else [],
    }


def parent(parent_id, child_ids, text=None, page=1, warnings=None):
    return {
        "parent_id": parent_id, "source": "doc.pdf", "page_start": page,
        "page_end": page + 1, "article_key": parent_id, "window_index": 1,
        "child_ids": child_ids, "text": text or f"parent text {parent_id}",
        "char_count": len(text or f"parent text {parent_id}"),
        "ambiguous_child_count": 0, "warnings": warnings or [],
    }


def hit(child_id, rank, queries=("Q0",), text=None):
    return {
        "child_id": child_id, "text": text or f"hit {child_id}",
        "source": "doc.pdf", "page_start": 1, "page_end": 1,
        "multi_query_rrf_score": 0.1, "multi_query_rank": rank,
        "support_query_count": len(queries), "support_query_ids": list(queries),
        "per_query_ranks": {query: rank for query in queries}, "per_query_trace": {},
    }


class ParentAggregationTests(unittest.TestCase):
    def test_child_maps_to_parent_and_formula_by_hand(self):
        config = cfg(parent_rrf_k=10)
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1), hit("c2", 3)],
            [child("c1", "p1"), child("c2", "p1")],
            [parent("p1", ["c1", "c2"])], config,
        )
        item = result["parents"][0]
        self.assertEqual("p1", item["parent_id"])
        self.assertAlmostEqual(1 / 11 + 1 / 13, item["parent_rrf_score"])
        self.assertEqual("c1", item["anchor_child_id"])
        self.assertEqual("p1", result["trace"]["child_to_parent_mapping"][0]["parent_id"])

    def test_score_cap_separates_scoring_and_supporting_children(self):
        config = cfg(parent_score_child_limit=2)
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1), hit("c2", 2), hit("c3", 3)],
            [child("c1", "p"), child("c2", "p"), child("c3", "p")],
            [parent("p", ["c1", "c2", "c3"])], config,
        )["parents"][0]
        self.assertEqual(["c1", "c2"], result["scoring_child_ids"])
        self.assertEqual(["c1", "c2", "c3"], result["supporting_child_ids"])

    def test_multiple_children_deduplicate_parent_and_union_queries(self):
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1, ("Q0", "Q1")), hit("c2", 2, ("Q1", "Q2"))],
            [child("c1", "p"), child("c2", "p")],
            [parent("p", ["c1", "c2"])], cfg(),
        )
        self.assertEqual(1, result["trace"]["unique_parent_count"])
        self.assertEqual(["Q0", "Q1", "Q2"], result["parents"][0]["support_query_ids"])

    def test_missing_child_or_parent_fails_with_id(self):
        with self.assertRaisesRegex(hr.HierarchyError, "missing"):
            hr.aggregate_parent_candidates([hit("missing", 1)], [], [], cfg())
        with self.assertRaisesRegex(hr.HierarchyError, "p-missing"):
            hr.aggregate_parent_candidates(
                [hit("c", 1)], [child("c", "p-missing")], [], cfg()
            )

    def test_sort_tie_break_and_candidate_limit(self):
        config = cfg(parent_candidates=1)
        result = hr.aggregate_parent_candidates(
            [hit("cb", 1), hit("ca", 1)],
            [child("cb", "pb"), child("ca", "pa")],
            [parent("pb", ["cb"]), parent("pa", ["ca"])], config,
        )
        self.assertEqual(["pa"], [item["parent_id"] for item in result["parents"]])
        self.assertEqual(["pb"], result["trace"]["parents_dropped_candidate_limit"])

    def test_context_budget_drops_only_whole_parent(self):
        config = cfg(total_context_max_chars=12)
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1), hit("c2", 2)],
            [child("c1", "p1"), child("c2", "p2")],
            [parent("p1", ["c1"], "12345678"), parent("p2", ["c2"], "abcdefgh")],
            config,
        )
        self.assertEqual(["p1"], [item["parent_id"] for item in result["parents"]])
        self.assertEqual(["p2"], result["trace"]["parents_dropped_context_budget"])
        self.assertEqual("12345678", result["parents"][0]["text"])

    def test_oversized_first_parent_is_kept_with_warning(self):
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1)], [child("c1", "p1")],
            [parent("p1", ["c1"], "x" * 30)], cfg(total_context_max_chars=10),
        )
        self.assertEqual(1, len(result["parents"]))
        self.assertIn("oversized_first_parent_context_budget", result["warnings"])

    def test_trace_expansion_ambiguity_and_no_reranker_generation(self):
        result = hr.aggregate_parent_candidates(
            [hit("c1", 1, text="12345")], [child("c1", "p", ambiguous=True)],
            [parent("p", ["c1"], "x" * 20)], cfg(),
        )
        self.assertEqual(5, result["trace"]["child_chars"])
        self.assertEqual(20, result["trace"]["expanded_parent_chars"])
        self.assertEqual(4.0, result["trace"]["context_expansion_factor"])
        self.assertEqual(1, result["trace"]["ambiguous_parent_count"])
        self.assertNotIn("rerank_score", result["parents"][0])
        self.assertNotIn("answer", result)

    def test_duplicate_child_text_across_parents_is_invariant_error(self):
        with self.assertRaisesRegex(hr.HierarchyError, "Duplicate child text"):
            hr.aggregate_parent_candidates(
                [], [child("c1", "p1", "same"), child("c2", "p2", "same")],
                [parent("p1", ["c1"]), parent("p2", ["c2"])], cfg(),
            )


class StoreAndPipelineTests(unittest.TestCase):
    def _write_store(self, root: Path, config, stale=False):
        input_dir = root / "input"
        store = root / "store"
        input_dir.mkdir(); store.mkdir()
        (input_dir / "chunks.json").write_text("[]", encoding="utf-8")
        children = [child("c1", "p1")]
        parents = [parent("p1", ["c1"], "expanded parent")]
        manifest = {
            "schema_version": hr.SCHEMA_VERSION, "strategy": "hierarchical",
            "config_identity": "stale" if stale else hr._config_identity(config),
            "input_fingerprints": hr.input_fingerprints(input_dir),
        }
        for name, value in (("children.json", children), ("parents.json", parents),
                            ("manifest.json", manifest)):
            (store / name).write_text(json.dumps(value), encoding="utf-8")
        return input_dir, store

    def test_missing_and_stale_store_return_hierarchy_not_ready(self):
        config = cfg()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = hr.load_hierarchy_store(config, root / "missing", root / "input")
            self.assertEqual("hierarchy_not_ready", missing["status"])
            input_dir, store = self._write_store(root, config, stale=True)
            stale = hr.load_hierarchy_store(config, store, input_dir)
            self.assertEqual("hierarchy_not_ready", stale["status"])
            self.assertIn("config_identity", stale["error"])

    def test_single_parent_pipeline_calls_hybrid_once_without_generator(self):
        config = cfg()
        with tempfile.TemporaryDirectory() as temp:
            input_dir, store = self._write_store(Path(temp), config)
            retriever = Mock(return_value={
                "candidates": [{
                    "chunk_id": "c1", "text": "hit", "source": "doc.pdf",
                    "page_start": 1, "page_end": 1, "bm25_rank": 1,
                    "bm25_score": 1.0, "semantic_rank": 1,
                    "semantic_distance": 0.1, "rrf_score": 0.03, "fused_rank": 1,
                }], "trace": {},
            })
            generator = Mock(side_effect=AssertionError("generator must not run"))
            result = hr.retrieve_parent_context(
                "câu hỏi", "single_parent", config, object(), [], generator,
                retriever, store, input_dir,
            )
            self.assertEqual("ready", result["status"])
            self.assertEqual("expanded parent", result["parents"][0]["text"])
            retriever.assert_called_once()
            generator.assert_not_called()


if __name__ == "__main__":
    unittest.main()
