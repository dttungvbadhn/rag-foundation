"""Kiểm thử offline BM25 lexical retrieval Buổi 08."""

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from rag_foundation.buoi_08 import advanced_rag
from rag_foundation.buoi_08 import rag


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chunks_advanced_sample.json"


class BM25Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, _ = rag.load_chunks(FIXTURE, strategy="hierarchical")

    def test_tokenizer_keeps_vietnamese_diacritics(self):
        tokens = advanced_rag.tokenize_vi_legal("cơ cấu lại thời hạn trả nợ")
        self.assertEqual(["cơ", "cấu", "lại", "thời", "hạn", "trả", "nợ"], tokens)

    def test_tokenizer_keeps_article_clause_numbers(self):
        tokens = advanced_rag.tokenize_vi_legal("Điều 7, Khoản 2")
        self.assertEqual(["điều", "7", "khoản", "2"], tokens)

    def test_corpus_and_query_use_same_tokenizer(self):
        original = advanced_rag.tokenize_vi_legal
        with patch.object(advanced_rag, "tokenize_vi_legal", wraps=original) as tokenizer:
            advanced_rag.bm25_search("Điều 7", self.chunks[:2], 2)
        inputs = [call.args[0] for call in tokenizer.call_args_list]
        self.assertIn("Điều 7", inputs)
        self.assertIn(self.chunks[0]["text"], inputs)
        self.assertIn(self.chunks[1]["text"], inputs)

    def test_exact_legal_term_ranks_above_unrelated_chunk(self):
        results = advanced_rag.bm25_search("Khoản 2 Điều 7 bổ sung hồ sơ", self.chunks, 8)
        rank = {item["chunk_id"]: item["bm25_rank"] for item in results}
        self.assertLess(rank["adv_h_003"], rank["adv_h_008"])

    def test_candidate_k_larger_than_corpus_and_source_unchanged(self):
        source = deepcopy(self.chunks[:3])
        results = advanced_rag.bm25_search("hồ sơ", source, 99)
        self.assertEqual(3, len(results))
        self.assertEqual(self.chunks[:3], source)

    def test_empty_or_tokenless_question_fails(self):
        for question in ("", "   ", "--- !!!"):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    advanced_rag.bm25_search(question, self.chunks, 3)

    def test_zero_score_candidates_are_kept_and_tie_break_by_chunk_id(self):
        chunks = [deepcopy(self.chunks[1]), deepcopy(self.chunks[0])]
        results = advanced_rag.bm25_search("từkhóakhôngtồntại", chunks, 2)
        self.assertEqual(2, len(results))
        self.assertEqual(sorted(item["chunk_id"] for item in chunks),
                         [item["chunk_id"] for item in results])
        self.assertTrue(all(item["bm25_score"] == 0.0 for item in results))

    def test_bm25_does_not_call_gemini_chroma_or_reranker(self):
        with patch.object(rag.genai, "Client") as gemini, patch.object(
            rag.chromadb, "PersistentClient"
        ) as chroma:
            results = advanced_rag.bm25_search("Điều 7", self.chunks, 3)
        self.assertEqual(3, len(results))
        gemini.assert_not_called()
        chroma.assert_not_called()
        self.assertNotIn("reranker_score", results[0])
        self.assertNotIn("distance", results[0])


if __name__ == "__main__":
    unittest.main()
