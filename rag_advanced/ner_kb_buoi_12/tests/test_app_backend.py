from __future__ import annotations

import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app_backend import (  # noqa: E402
    csv_statistics,
    document_bundle,
    dot_graph,
    load_app_data,
    search_documents,
)


class AppBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_app_data(BASE_DIR)

    def test_statistics_match_validated_outputs(self) -> None:
        stats = csv_statistics(self.data)
        self.assertEqual(stats["documents"], 30)
        self.assertEqual(stats["entities"], 97)
        self.assertEqual(stats["relationships"], 187)
        self.assertEqual(stats["validation"].get("PASS"), 187)

    def test_search_by_document_number(self) -> None:
        result = search_documents(self.data.documents, "22/2023/TT-NHNN")
        self.assertFalse(result.empty)
        self.assertIn("22/2023/TT-NHNN", set(result["so_ky_hieu"]))

    def test_document_bundle_has_traceability(self) -> None:
        document_id = str(self.data.documents.iloc[0]["id"])
        bundle = document_bundle(self.data, document_id)
        self.assertEqual(str(bundle["document"]["id"]), document_id)
        self.assertFalse(bundle["entities"].empty)
        self.assertTrue(bundle["entities"]["evidence"].fillna("").str.strip().ne("").all())

    def test_dot_graph_escapes_labels(self) -> None:
        dot = dot_graph(
            [{"id": "doc-1", "label": 'Văn bản "mẫu"', "type": "Document"}],
            [],
        )
        self.assertIn("digraph KnowledgeGraph", dot)
        self.assertIn('\\"mẫu\\"', dot)


if __name__ == "__main__":
    unittest.main()
