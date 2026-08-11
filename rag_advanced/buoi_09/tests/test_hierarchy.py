"""Offline tests cho deterministic hierarchy registry và parent store."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rag_advanced.buoi_09 import hierarchical_rag as hr


def chunk(child_id, text, source="a.pdf", page=1, structure=None):
    value = {
        "chunk_id": child_id, "strategy": "hierarchical", "source": source,
        "page_start": page, "page_end": page, "text": text,
    }
    if structure is not None:
        value["structure"] = structure
    return value


class ResolutionTests(unittest.TestCase):
    def test_metadata_precedence_and_conflict_warning(self):
        children = hr.resolve_hierarchy([
            chunk("x_1", "Điều 8. Heading khác", structure={"article": "Điều 7"})
        ])
        self.assertEqual("Điều 7", children[0]["structural_path"]["article"])
        self.assertEqual("metadata", children[0]["resolution_method"])
        self.assertTrue(children[0]["ambiguous"])
        self.assertIn("metadata_heading_conflict:article", children[0]["warnings"])

    def test_multiple_article_headings_are_ambiguous(self):
        child = hr.resolve_hierarchy([
            chunk("x_1", "Điều 7. Một\nĐiều 8. Hai\nNội dung")
        ])[0]
        self.assertTrue(child["ambiguous"])
        self.assertIn("multiple_heading_candidates:article", child["warnings"])
        self.assertEqual("Điều 7", child["structural_path"]["article"])

    def test_heading_inferred_only_at_start_inline_reference_ignored(self):
        children = hr.resolve_hierarchy([
            chunk("x_1", "Điều 7. Nội dung"),
            chunk("x_2", "Theo khoản 4 Điều 99, báo cáo được thực hiện."),
        ])
        self.assertEqual("heading_inferred", children[0]["resolution_method"])
        self.assertEqual("Điều 7", children[1]["structural_path"]["article"])
        self.assertEqual("carried_forward", children[1]["resolution_method"])

    def test_carry_forward_never_crosses_source(self):
        children = hr.resolve_hierarchy([
            chunk("a_1", "Điều 2. A", source="a.pdf"),
            chunk("a_2", "Nội dung tiếp", source="a.pdf"),
            chunk("b_1", "Nội dung B", source="b.pdf"),
        ])
        by_id = {item["child_id"]: item for item in children}
        self.assertEqual("Điều 2", by_id["a_2"]["structural_path"]["article"])
        self.assertEqual("document_fallback", by_id["b_1"]["resolution_method"])
        self.assertIsNone(by_id["b_1"]["structural_path"]["article"])

    def test_numeric_order_and_duplicate(self):
        children = hr.resolve_hierarchy([
            chunk("x_10", "ten"), chunk("x_2", "two"), chunk("x_1", "one")
        ])
        self.assertEqual(["x_1", "x_2", "x_10"], [item["child_id"] for item in children])
        with self.assertRaisesRegex(hr.HierarchyError, "Duplicate"):
            hr.resolve_hierarchy([chunk("x_1", "a"), chunk("x_1", "b")])


class ParentTests(unittest.TestCase):
    def test_stable_id_one_parent_per_child_pages_and_text(self):
        raw = [chunk("x_1", "Điều 1. A", page=3), chunk("x_2", "B", page=5)]
        first_children, first_parents = hr.build_parents(hr.resolve_hierarchy(raw), 1000)
        second_children, second_parents = hr.build_parents(hr.resolve_hierarchy(raw), 1000)
        self.assertEqual(first_parents[0]["parent_id"], second_parents[0]["parent_id"])
        self.assertEqual(1, len({item["parent_id"] for item in first_children}))
        self.assertEqual(["x_1", "x_2"], first_parents[0]["child_ids"])
        self.assertEqual((3, 5), (first_parents[0]["page_start"], first_parents[0]["page_end"]))
        self.assertEqual("Điều 1. A\n\nB", first_parents[0]["text"])
        self.assertEqual(len(first_parents[0]["text"]), first_parents[0]["char_count"])

    def test_split_at_child_boundary(self):
        raw = [chunk(f"x_{index}", ("Điều 1. " if index == 1 else "") + "a" * 590)
               for index in range(1, 4)]
        children, parents = hr.build_parents(hr.resolve_hierarchy(raw), 1000)
        self.assertEqual(3, len(parents))
        self.assertEqual(3, len({child["parent_id"] for child in children}))
        self.assertEqual([["x_1"], ["x_2"], ["x_3"]], [p["child_ids"] for p in parents])

    def test_oversized_single_child_is_not_truncated(self):
        text = "Điều 1. " + "x" * 1100
        _, parents = hr.build_parents(hr.resolve_hierarchy([chunk("x_1", text)]), 1000)
        self.assertEqual(text, parents[0]["text"])
        self.assertIn("oversized_single_child", parents[0]["warnings"])


class StoreTests(unittest.TestCase):
    def test_status_cli_parser(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(hr, "HIERARCHY_STORE", Path(temporary) / "missing"), \
             patch("sys.argv", ["hierarchical_rag.py", "hierarchy-status"]):
            self.assertEqual(0, hr.main())

    def test_atomic_write_manifest_fingerprint_and_status_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"; input_dir.mkdir()
            fixture = input_dir / "chunks.json"
            fixture.write_text(json.dumps([chunk("x_1", "Điều 1. A")]), encoding="utf-8")
            config = hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH)
            registry = hr.build_registry(config, input_dir)
            expected = hashlib.sha256(fixture.read_bytes()).hexdigest()
            self.assertEqual(expected, registry["manifest"]["input_fingerprints"][0]["sha256"])
            store = root / "store"
            hr.write_registry(registry, store)
            self.assertEqual({"children.json", "parents.json", "manifest.json"},
                             {path.name for path in store.iterdir()})
            before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in store.iterdir()}
            time.sleep(0.01)
            status = hr.hierarchy_status(store)
            after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in store.iterdir()}
            self.assertTrue(status["complete"])
            self.assertEqual(before, after)

    def test_status_missing_store_does_not_create_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            status = hr.hierarchy_status(missing)
            self.assertFalse(status["store_exists"])
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
