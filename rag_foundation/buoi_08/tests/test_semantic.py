"""Kiểm thử offline semantic candidate stage Buổi 08."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag_foundation.buoi_08 import advanced_rag
from rag_foundation.buoi_08 import rag


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "chunks_advanced_sample.json"


class Embedding:
    def __init__(self, values):
        self.values = values


class EmbedResponse:
    def __init__(self, values):
        self.embeddings = [Embedding(values)]


class FakeModels:
    def __init__(self):
        self.embed_calls = []
        self.generate_content = Mock(side_effect=AssertionError("generation không được gọi"))

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        seed = (sum(ord(char) for char in kwargs["contents"]) % 89 + 1) / 100
        return EmbedResponse([seed] + [0.0] * 127)


class FakeGemini:
    def __init__(self):
        self.models = FakeModels()


def config(api_key="test-key"):
    return advanced_rag.AdvancedConfig(
        api_key=api_key,
        embedding_model="model-test",
        embedding_dim=128,
        generation_model="generation-test",
        max_distance=0.45,
        bm25_candidates=20,
        semantic_candidates=20,
        rrf_k=60,
        rrf_bm25_weight=1.0,
        rrf_semantic_weight=1.0,
        rerank_candidates=20,
        final_top_k=5,
        reranker_model="reranker-test",
        reranker_max_length=512,
        rerank_batch_size=4,
        rerank_min_score=0.5,
        rerank_device="cpu",
    )


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.storage = Path(self.temporary.name) / "chroma"
        self.client = None

    def tearDown(self):
        if self.client is not None:
            self.client._system.stop()
        rag.chromadb.api.client.SharedSystemClient.clear_system_cache()
        self.temporary.cleanup()

    def _prepare(self):
        self.client = rag.chromadb.PersistentClient(path=self.storage)
        gemini = FakeGemini()
        advanced_rag.prepare_semantic(
            "hierarchical",
            config(),
            input_path=FIXTURE,
            storage_path=self.storage,
            embedding_client=gemini,
            chroma_client=self.client,
        )
        return gemini

    def test_semantic_top_k_count_order_metadata_and_no_generation(self):
        gemini = self._prepare()
        candidates = advanced_rag.semantic_candidates(
            "Điều 7 quy định gì?",
            3,
            "hierarchical",
            config(),
            storage_path=self.storage,
            embedding_client=gemini,
            chroma_client=self.client,
        )
        self.assertEqual(3, len(candidates))
        self.assertEqual([1, 2, 3], [item["semantic_rank"] for item in candidates])
        self.assertEqual(sorted(item["semantic_distance"] for item in candidates),
                         [item["semantic_distance"] for item in candidates])
        for item in candidates:
            self.assertEqual(
                {"chunk_id", "text", "source", "page_start", "page_end",
                 "semantic_rank", "semantic_distance"},
                set(item),
            )
        gemini.models.generate_content.assert_not_called()

    def test_candidate_k_above_count_uses_collection_count(self):
        gemini = self._prepare()
        candidates = advanced_rag.semantic_candidates(
            "hồ sơ", 99, "hierarchical", config(), storage_path=self.storage,
            embedding_client=gemini, chroma_client=self.client
        )
        self.assertEqual(8, len(candidates))

    def test_collection_metadata_mismatch_is_blocked(self):
        gemini = self._prepare()
        name = rag.collection_name("hierarchical", "model-test", 128)
        collection = self.client.get_collection(name, embedding_function=None)
        collection.modify(metadata={**collection.metadata, "embedding_dim": 256})
        with self.assertRaisesRegex(rag.RagIndexError, "--reset"):
            advanced_rag.semantic_candidates(
                "hồ sơ", 3, "hierarchical", config(), storage_path=self.storage,
                embedding_client=gemini, chroma_client=self.client
            )

    def test_status_on_empty_storage_does_not_create_collection_or_call_api(self):
        with patch.object(rag.chromadb, "PersistentClient") as persistent, patch.object(
            rag.genai, "Client"
        ) as gemini:
            result = advanced_rag.advanced_status(
                "hierarchical", config(), input_path=FIXTURE, storage_path=self.storage
            )
        self.assertTrue(result["bm25_ready"])
        self.assertEqual(8, result["corpus_size"])
        self.assertFalse(result["semantic_collection_exists"])
        self.assertFalse(self.storage.exists())
        persistent.assert_not_called()
        gemini.assert_not_called()

    def test_missing_key_does_not_create_embedding_or_chroma(self):
        embedding = Mock()
        chroma = Mock()
        with self.assertRaisesRegex(rag.RagIndexError, "Thiếu GEMINI_API_KEY"):
            advanced_rag.prepare_semantic(
                "hierarchical", config(api_key=""), input_path=FIXTURE,
                storage_path=self.storage, embedding_client=embedding, chroma_client=chroma
            )
        embedding.assert_not_called()
        chroma.assert_not_called()


if __name__ == "__main__":
    unittest.main()
