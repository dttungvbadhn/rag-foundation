import unittest

from pipeline import chunk_document, clean_text


class PipelineTests(unittest.TestCase):
    def test_clean_preserves_legal_terms(self):
        result = clean_text("<p>Căn cứ Luật số 01/2020/QH14</p><p>thay thế văn bản cũ</p>")
        self.assertIn("Căn cứ", result)
        self.assertIn("01/2020/QH14", result)
        self.assertIn("thay thế", result)

    def test_hierarchy_and_ids_are_stable(self):
        chunks = chunk_document("42", "<h1>Chương I</h1><h2>Điều 1</h2><p>Nội dung</p>")
        self.assertEqual(["42:chunk:00000", "42:chunk:00001", "42:chunk:00002"], [c.id for c in chunks])
        self.assertEqual(chunks[1].id, chunks[2].parent_id)


if __name__ == "__main__":
    unittest.main()
