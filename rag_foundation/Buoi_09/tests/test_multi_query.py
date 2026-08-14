"""Multi-query generator tests hoàn toàn offline."""

from dataclasses import replace
import unicodedata
import unittest
from unittest.mock import Mock

from rag_foundation.Buoi_09 import hierarchical_rag as hr


def config():
    return hr.load_hierarchy_config(hr.ENV_EXAMPLE_PATH)


def payload(*items):
    return {"queries": [{"text": text, "focus": focus} for text, focus in items]}


class MultiQueryTests(unittest.TestCase):
    def setUp(self):
        hr.clear_query_cache()

    def test_q0_is_first_trimmed_nfc_and_ids_deterministic(self):
        decomposed = "  Diều kiện vay vốn?  "
        fake = Mock(return_value=payload(
            ("Thuật ngữ điều kiện vay vốn", "exact_legal_terms"),
            ("Yêu cầu để được vay vốn", "paraphrase"),
        ))
        result = hr.generate_query_set(decomposed, config(), fake)
        self.assertEqual("ready", result["status"])
        self.assertEqual(unicodedata.normalize("NFC", decomposed).strip(), result["queries"][0]["text"])
        self.assertEqual("original", result["queries"][0]["origin"])
        self.assertEqual(["Q0", "Q1", "Q2"], [item["query_id"] for item in result["queries"]])
        fake.assert_called_once()

    def test_strict_schema_rejects_extra_answer_or_bad_focus(self):
        for invalid in (
            {"queries": [], "answer": "không được phép"},
            {"queries": [{"text": "x", "focus": "answer"}]},
            {"queries": "not-list"},
        ):
            hr.clear_query_cache()
            result = hr.generate_query_set("Câu hỏi", config(), Mock(return_value=invalid))
            self.assertEqual("query_generation_unavailable", result["status"])
            self.assertEqual(["Q0"], [item["query_id"] for item in result["queries"]])

    def test_duplicate_removal_normalizes_case_whitespace_punctuation(self):
        fake = Mock(return_value=payload(
            ("CÂU   HỎI!!!", "paraphrase"),
            ("câu hỏi", "exact_legal_terms"),
            ("khía cạnh khác", "missing_aspect"),
        ))
        result = hr.generate_query_set("Câu hỏi", config(), fake)
        self.assertEqual("ready", result["status"])
        self.assertEqual(2, len(result["queries"]))
        self.assertEqual(2, result["dropped_duplicate_count"])

    def test_max_length_skips_invalid_but_keeps_valid(self):
        cfg = replace(config(), multi_query_max_chars=50)
        result = hr.generate_query_set("Câu hỏi", cfg, Mock(return_value=payload(
            ("x" * 51, "paraphrase"), ("truy vấn hợp lệ", "missing_aspect")
        )))
        self.assertEqual("ready", result["status"])
        self.assertEqual("truy vấn hợp lệ", result["queries"][1]["text"])

    def test_reference_preserved_and_invented_article_dropped(self):
        result = hr.generate_query_set(
            "Khoản 2 Điều 7 quy định gì năm 2023?", config(),
            Mock(return_value=payload(
                ("Điều 99 quy định nội dung gì?", "exact_legal_terms"),
                ("Nội dung Khoản 2 Điều 7 áp dụng năm 2023", "paraphrase"),
            )),
        )
        self.assertEqual("ready", result["status"])
        self.assertEqual(1, result["dropped_invalid_reference_count"])
        self.assertIn("Khoản 2 Điều 7", result["queries"][1]["text"])

    def test_missing_original_reference_is_explicit_failure(self):
        result = hr.generate_query_set(
            "Điều 7 quy định gì?", config(),
            Mock(return_value=payload(("Quy định này có nội dung gì?", "paraphrase"))),
        )
        self.assertEqual("query_generation_unavailable", result["status"])
        self.assertIn("legal reference", result["error"])

    def test_cache_hit_does_not_call_generator_twice(self):
        fake = Mock(return_value=payload(("truy vấn khác", "paraphrase")))
        first = hr.generate_query_set("Câu hỏi cache", config(), fake)
        second = hr.generate_query_set("Câu hỏi cache", config(), fake)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        fake.assert_called_once()

    def test_generator_error_is_explicit_and_no_network(self):
        fake = Mock(side_effect=OSError("offline"))
        result = hr.generate_query_set("Câu hỏi", config(), fake)
        self.assertEqual("query_generation_unavailable", result["status"])
        self.assertEqual("Q0", result["queries"][0]["query_id"])
        self.assertIn("OSError", result["error"])
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
